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
    auto globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);
    auto edges = std::get<0>(globalEdgeIndex);

    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>(
        edges.size(), edges.size(), stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>(
        edges.size(), edges.size(), stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>(
        edges.size(), edges.size(), stream);
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<float>>(
        this->mesh->elements.rows(), math::square(3), stream, 0.0, false);

    this->initElementalMatrices(stream);
    this->initJacobianCalculationMatrix(stream);
}

// init elemental matrices
void mpFlow::MWI::Equation::initElementalMatrices(cudaStream_t stream) {
    // calculate indices of unique mesh edges
    auto globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);
    auto edges = std::get<0>(globalEdgeIndex);

    // create intermediate matrices
    auto elementCount = std::make_shared<numeric::SparseMatrix<unsigned>>(
        edges.size(), edges.size(), stream);
    std::vector<std::shared_ptr<numeric::SparseMatrix<unsigned>>> connectivityMatrices;
    std::vector<std::shared_ptr<numeric::SparseMatrix<thrust::complex<float>>>> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];

        // extract coordinats of node points of element
        Eigen::ArrayXXd points = this->mesh->elementNodes(element);

        // set connectivity and elemental residual matrix elements
        for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            size_t level = elementCount->getValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::make_shared<numeric::SparseMatrix<unsigned>>(
                    edges.size(), edges.size(), stream));
                elementalSMatrices.push_back(std::make_shared<numeric::SparseMatrix<thrust::complex<float>>>(
                    edges.size(), edges.size(), stream));
                elementalRMatrices.push_back(std::make_shared<numeric::SparseMatrix<thrust::complex<float>>>(
                    edges.size(), edges.size(), stream));
            }

            // set connectivity element
            connectivityMatrices[level]->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]),
                element);

            // evaluate integral equations
            Eigen::ArrayXi edgeI(2), edgeJ(2);
            edgeI << std::get<0>(std::get<1>(localEdges[i])), std::get<1>(std::get<1>(localEdges[i]));
            edgeJ << std::get<0>(std::get<1>(localEdges[j])), std::get<1>(std::get<1>(localEdges[j]));
            auto basisI = std::make_shared<FEM::basis::Edge>(points, edgeI);
            auto basisJ = std::make_shared<FEM::basis::Edge>(points, edgeJ);

            // set elemental system element
            elementalSMatrices[level]->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]),
                basisI->integrateGradientWithBasis(basisJ));

            // set elemental residual element
            elementalRMatrices[level]->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]),
                basisI->integrateWithBasis(basisJ));

            // increment element count
            elementCount->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]),
                elementCount->getValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j])) + 1);
        }
    }

    // determine nodes with common element
    auto commonElementMatrix = std::make_shared<numeric::SparseMatrix<thrust::complex<float>>>(
        edges.size(), edges.size(), stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];

        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            commonElementMatrix->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]), 1.0f);
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // fill sparse matrices initially with commonElementMatrix
    this->sMatrix->copy(commonElementMatrix, stream);
    this->rMatrix->copy(commonElementMatrix, stream);
    this->systemMatrix->copy(commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<unsigned>>(
        edges.size(), numeric::sparseMatrix::blockSize * connectivityMatrices.size(),
        stream, constants::invalidIndex);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<thrust::complex<float>>>(edges.size(),
        numeric::sparseMatrix::blockSize * elementalSMatrices.size(), stream);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<thrust::complex<float>>>(edges.size(),
        numeric::sparseMatrix::blockSize * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (unsigned level = 0; level < connectivityMatrices.size(); ++level) {
        for (int element = 0; element < this->mesh->elements.rows(); ++element) {
            auto localEdges = std::get<1>(globalEdgeIndex)[element];

            for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < 3; ++j) {
                unsigned columId = commonElementMatrix->getColumnId(std::get<0>(localEdges[i]),
                    std::get<0>(localEdges[j]));

                (*this->connectivityMatrix)(std::get<0>(localEdges[i]), level * numeric::sparseMatrix::blockSize + columId) =
                    connectivityMatrices[level]->getValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
                (*this->elementalSMatrix)(std::get<0>(localEdges[i]), level * numeric::sparseMatrix::blockSize + columId) =
                    elementalSMatrices[level]->getValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
                (*this->elementalRMatrix)(std::get<0>(localEdges[i]), level * numeric::sparseMatrix::blockSize + columId) =
                    elementalRMatrices[level]->getValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
            }
        }
    }
    this->connectivityMatrix->copyToDevice(stream);
    this->elementalSMatrix->copyToDevice(stream);
    this->elementalRMatrix->copyToDevice(stream);

    // update sMatrix only once
    auto alpha = std::make_shared<numeric::Matrix<thrust::complex<float>>>(this->mesh->elements.rows(), 1, stream);
    FEM::equation::updateMatrix<thrust::complex<float>, false>(this->elementalSMatrix, alpha, this->connectivityMatrix,
        thrust::complex<float>(1.0, 0.0), stream, this->sMatrix);
}

void mpFlow::MWI::Equation::initJacobianCalculationMatrix(cudaStream_t stream) {
    // calculate indices of unique mesh edges
    auto globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);
    auto edges = std::get<0>(globalEdgeIndex);

    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<float>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];

        // extract coordinats of node points of element
        Eigen::ArrayXXd points = this->mesh->elementNodes(element);

        // fill matrix
        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            // evaluate integral equations
            Eigen::ArrayXi edgeI(2), edgeJ(2);
            edgeI << std::get<0>(std::get<1>(localEdges[i])), std::get<1>(std::get<1>(localEdges[i]));
            edgeJ << std::get<0>(std::get<1>(localEdges[j])), std::get<1>(std::get<1>(localEdges[j]));
            auto basisI = std::make_shared<FEM::basis::Edge>(points, edgeI);
            auto basisJ = std::make_shared<FEM::basis::Edge>(points, edgeJ);

            // set elementalJacobianMatrix element
            (*elementalJacobianMatrix)(element, i + j * 3) =
                basisI->integrateGradientWithBasis(basisJ);
        }
    }

    elementalJacobianMatrix->copyToDevice(stream);
    this->elementalJacobianMatrix->copy(elementalJacobianMatrix, stream);
}


void mpFlow::MWI::Equation::update(const std::shared_ptr<numeric::Matrix<thrust::complex<float>>> beta,
    thrust::complex<float> k, cudaStream_t stream) {
    // check input
    if (beta == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Equation::update: beta == nullptr");
    }

    // update matrices
    auto alpha = std::make_shared<numeric::Matrix<thrust::complex<float>>>(beta->rows, beta->cols, stream,
        1.0f);
    FEM::equation::updateMatrix<thrust::complex<float>, false>(this->elementalSMatrix, alpha,
        this->connectivityMatrix, thrust::complex<float>(1.0, 0.0), stream, this->sMatrix);
    FEM::equation::updateMatrix<thrust::complex<float>, false>(this->elementalRMatrix, beta,
        this->connectivityMatrix, thrust::complex<float>(1.0, 0.0), stream, this->rMatrix);

    // update system matrix
    FEM::equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::blockSize,
        numeric::matrix::blockSize, stream, this->sMatrix->deviceValues, this->rMatrix->deviceValues,
        this->sMatrix->deviceColumnIds, this->sMatrix->density, k, this->systemMatrix->deviceValues);
}
