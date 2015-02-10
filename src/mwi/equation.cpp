// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"
#include "mpflow/fem/equation_kernel.h"

mpFlow::MWI::Equation::Equation(std::shared_ptr<numeric::IrregularMesh> mesh,
    cudaStream_t stream)
    : mesh(mesh) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Equation::Equation: mesh == nullptr");
    }

    // init matrices
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh->elements->rows, math::square(3), stream, 0.0, false);

    this->initElementalMatrices(stream);
    this->initJacobianCalculationMatrix(stream);
}

// init elemental matrices
void mpFlow::MWI::Equation::initElementalMatrices(cudaStream_t stream) {
    // calculate indices of unique mesh edges
    auto globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);
    auto edges = std::get<0>(globalEdgeIndex);

    // create intermediate matrices
    Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic> elementCount =
        Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>::Zero(this->mesh->nodes->rows,
        this->mesh->nodes->rows);
    std::vector<Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>> connectivityMatrices;
    std::vector<Eigen::Array<dtype::real, Eigen::Dynamic, Eigen::Dynamic>> elementalRMatrices;
    auto sMatrix = std::make_shared<numeric::Matrix<dtype::complex>>(edges.size(), edges.size(), stream);

    // fill intermediate connectivity and elemental matrices
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];
        auto points = std::get<1>(mesh->elementNodes(element));

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < 3; i++)
        for (dtype::index j = 0; j < 3; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            size_t level = elementCount(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>
                    ::Ones(this->mesh->nodes->rows, this->mesh->nodes->rows)
                    * dtype::invalid_index);
                elementalRMatrices.push_back(Eigen::Array<dtype::real, Eigen::Dynamic, Eigen::Dynamic>
                    ::Zero(this->mesh->nodes->rows, this->mesh->nodes->rows));
            }

            // set connectivity element
            connectivityMatrices[level](std::get<0>(localEdges[i]), std::get<0>(localEdges[j])) =
                element;

            // evaluate integral equations
            auto edgeI = std::make_shared<FEM::basis::Edge>(points, std::get<1>(localEdges[i]));
            auto edgeJ = std::make_shared<FEM::basis::Edge>(points, std::get<1>(localEdges[j]));

            (*sMatrix)(std::get<0>(localEdges[i]), std::get<0>(localEdges[j])) +=
                edgeI->integrateGradientWithBasis(edgeJ);
            elementalRMatrices[level](std::get<0>(localEdges[i]), std::get<0>(localEdges[j])) =
                edgeI->integrateWithBasis(edgeJ);

            // increment element count
            elementCount(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]))++;
        }
    }
    sMatrix->copyToDevice(stream);

    // create sparse matrices
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::complex>>(
        sMatrix, stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::complex>>(
        sMatrix, stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::complex>>(
        sMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = FEM::equation::reduceMatrix<dtype::index>(
        connectivityMatrices, this->sMatrix, stream);
    this->elementalRMatrix = FEM::equation::reduceMatrix<dtype::complex>(
        elementalRMatrices, this->sMatrix, stream);

    cudaStreamSynchronize(stream);
}

void mpFlow::MWI::Equation::initJacobianCalculationMatrix(cudaStream_t stream) {
    // calculate indices of unique mesh edges
    auto localEdgeIndices = std::get<1>(numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements));

    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        auto localEdges = localEdgeIndices[element];
        auto points = std::get<1>(mesh->elementNodes(element));

        // fill matrix
        for (dtype::index i = 0; i < 3; ++i)
        for (dtype::index j = 0; j < 3; ++j) {
            // evaluate integral equations
            auto edgeI = std::make_shared<FEM::basis::Edge>(points, std::get<1>(localEdges[i]));
            auto edgeJ = std::make_shared<FEM::basis::Edge>(points, std::get<1>(localEdges[j]));

            // set elementalJacobianMatrix element
            (*elementalJacobianMatrix)(element, i + j * 3) =
                edgeI->integrateGradientWithBasis(edgeJ);
        }
    }

    elementalJacobianMatrix->copyToDevice(stream);
    this->elementalJacobianMatrix->copy(elementalJacobianMatrix, stream);
}


void mpFlow::MWI::Equation::update(const std::shared_ptr<numeric::Matrix<dtype::complex>> beta,
    dtype::complex k, cudaStream_t stream) {
    // check input
    if (beta == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Equation::update: beta == nullptr");
    }

    // update matrices
    FEM::equation::updateMatrix(this->elementalRMatrix, beta, this->connectivityMatrix,
        dtype::complex(1.0, 0.0), stream, this->rMatrix);

    // update system matrix
    FEM::equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix->values, this->rMatrix->values,
        this->sMatrix->columnIds, this->sMatrix->density, k, this->systemMatrix->values);
}
