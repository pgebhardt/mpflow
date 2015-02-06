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
    Eigen::ArrayXXi elementCount = Eigen::ArrayXXi::Zero(edges.size(), edges.size());
    std::array<Eigen::ArrayXXf, 2> connectivityMatrices = {{
        Eigen::ArrayXXf::Zero(edges.size(), edges.size()), Eigen::ArrayXXf::Zero(edges.size(), edges.size()) }};
    auto sMatrix = std::make_shared<numeric::Matrix<dtype::real>>(edges.size(), edges.size(), stream);
    std::array<Eigen::ArrayXXf, 2> elementalRMatrices = {{
        Eigen::ArrayXXf::Zero(edges.size(), edges.size()), Eigen::ArrayXXf::Zero(edges.size(), edges.size()) }};

    // fill intermediate connectivity and elemental matrices
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];

        // extract coordinats of node points of element
        std::array<std::tuple<dtype::real, dtype::real>, 3> points;
        dtype::index i = 0;
        for (const auto& point : mesh->elementNodes(element)) {
            points[i] = std::get<1>(point);
            i++;
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < 3; i++)
        for (dtype::index j = 0; j < 3; j++) {
            // get current element count
            auto level = elementCount(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));

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
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        sMatrix, stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        sMatrix, stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        sMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        edges.size(),numeric::sparseMatrix::block_size * connectivityMatrices.size(),
        stream, dtype::invalid_index);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dtype::real>>(edges.size(),
        numeric::sparseMatrix::block_size * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    auto connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        connectivityMatrices[0].rows(), connectivityMatrices[0].cols(), stream,
        dtype::invalid_index);
    auto elementalRMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        elementalRMatrices[0].rows(), elementalRMatrices[0].cols(), stream);

    for (dtype::index level = 0; level < connectivityMatrices.size(); ++level) {
        for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
            auto localEdges = std::get<1>(globalEdgeIndex)[element];

            for (dtype::index i = 0; i < 3; ++i)
            for (dtype::index j = 0; j < 3; ++j) {
                (*connectivityMatrix)(std::get<0>(localEdges[i]), std::get<0>(localEdges[j])) =
                    connectivityMatrices[level](std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
                (*elementalRMatrix)(std::get<0>(localEdges[i]), std::get<0>(localEdges[j])) =
                    elementalRMatrices[level](std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
            }
        }
        connectivityMatrix->copyToDevice(stream);
        elementalRMatrix->copyToDevice(stream);
        cudaStreamSynchronize(stream);

        FEM::equation::reduceMatrix(connectivityMatrix, this->sMatrix, level, stream,
            this->connectivityMatrix);
        FEM::equation::reduceMatrix(elementalRMatrix, this->rMatrix, level, stream,
            this->elementalRMatrix);
    }
}

void mpFlow::MWI::Equation::initJacobianCalculationMatrix(cudaStream_t stream) {
    // calculate indices of unique mesh edges
    auto globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);
    auto edges = std::get<0>(globalEdgeIndex);

    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];

        // extract coordinats of node points of element
        std::array<std::tuple<dtype::real, dtype::real>, 3> points;
        dtype::index i = 0;
        for (const auto& point : mesh->elementNodes(element)) {
            points[i] = std::get<1>(point);
            i++;
        }

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


void mpFlow::MWI::Equation::update(const std::shared_ptr<numeric::Matrix<dtype::real>> beta,
    dtype::real k, cudaStream_t stream) {
    // check input
    if (beta == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Equation::update: beta == nullptr");
    }

    // update matrices
    FEM::equation::updateMatrix(this->elementalRMatrix, beta, this->connectivityMatrix,
        1.0f, stream, this->rMatrix);

    // update system matrix
    FEM::equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix->values, this->rMatrix->values,
        this->sMatrix->columnIds, this->sMatrix->density, k, this->systemMatrix->values);
}
