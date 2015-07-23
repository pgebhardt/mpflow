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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"
#include "mpflow/fem/equation_kernel.h"
#include "mpflow/mwi/equation_kernel.h"

template <
    class dataType
>
mpFlow::MWI::Equation<dataType>::Equation(std::shared_ptr<numeric::IrregularMesh const> const mesh,
    cudaStream_t const stream)
    : mesh(mesh) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Equation::Equation: mesh == nullptr");
    }

    // init matrices
    this->edges = numeric::Matrix<int>::fromEigen(std::get<0>(numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements)), stream);
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->edges->rows, this->edges->rows, stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->edges->rows, this->edges->rows, stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->edges->rows, this->edges->rows, stream);
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements.rows(), math::square(3), stream, 0.0, false);
    
    this->initElementalMatrices(stream);
    this->initJacobianCalculationMatrix(stream);
}

// init elemental matrices
template <
    class dataType
>
void mpFlow::MWI::Equation<dataType>::initElementalMatrices(cudaStream_t const stream) {
    // calculate indices of unique mesh edges
    auto const globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);

    // create intermediate matrices
    auto const elementCount = std::make_shared<numeric::SparseMatrix<unsigned>>(
        this->edges->rows, this->edges->rows, stream);
    std::vector<std::shared_ptr<numeric::SparseMatrix<unsigned>>> connectivityMatrices;
    std::vector<std::shared_ptr<numeric::SparseMatrix<dataType>>> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        auto const localEdges = std::get<1>(globalEdgeIndex)[element];

        // extract coordinats of node points of element
        auto const points = this->mesh->elementNodes(element);

        // set connectivity and elemental residual matrix elements
        for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            size_t const level = elementCount->getValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::make_shared<numeric::SparseMatrix<unsigned>>(
                    this->edges->rows, this->edges->rows, stream));
                elementalSMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->edges->rows, this->edges->rows, stream));
                elementalRMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->edges->rows, this->edges->rows, stream));
            }

            // set connectivity element
            connectivityMatrices[level]->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]),
                element);

            // evaluate integral equations
            Eigen::ArrayXi edgeI(2), edgeJ(2);
            edgeI << std::get<0>(std::get<1>(localEdges[i])), std::get<1>(std::get<1>(localEdges[i]));
            edgeJ << std::get<0>(std::get<1>(localEdges[j])), std::get<1>(std::get<1>(localEdges[j]));
            auto const basisI = std::make_shared<FEM::basis::Edge>(points, edgeI);
            auto const basisJ = std::make_shared<FEM::basis::Edge>(points, edgeJ);

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
    auto const commonElementMatrix = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->edges->rows, this->edges->rows, stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        auto localEdges = std::get<1>(globalEdgeIndex)[element];

        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            commonElementMatrix->setValue(std::get<0>(localEdges[i]), std::get<0>(localEdges[j]), dataType(1));
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // fill sparse matrices initially with commonElementMatrix
    this->sMatrix->copy(commonElementMatrix, stream);
    this->rMatrix->copy(commonElementMatrix, stream);
    this->systemMatrix->copy(commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<unsigned>>(
        this->edges->rows, numeric::sparseMatrix::blockSize * connectivityMatrices.size(),
        stream, constants::invalidIndex);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<dataType>>(this->edges->rows,
        numeric::sparseMatrix::blockSize * elementalSMatrices.size(), stream);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dataType>>(this->edges->rows,
        numeric::sparseMatrix::blockSize * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (unsigned level = 0; level < connectivityMatrices.size(); ++level) {
        for (int element = 0; element < this->mesh->elements.rows(); ++element) {
            auto const localEdges = std::get<1>(globalEdgeIndex)[element];

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
    auto const alpha = std::make_shared<numeric::Matrix<dataType>>(this->mesh->elements.rows(), 1, stream);
    FEM::equation::updateMatrix<dataType, false>(this->elementalSMatrix, alpha, this->connectivityMatrix,
        dataType(1), stream, this->sMatrix);
}

template <
    class dataType
>
void mpFlow::MWI::Equation<dataType>::initJacobianCalculationMatrix(cudaStream_t const stream) {
    // calculate indices of unique mesh edges
    auto const globalEdgeIndex = numeric::irregularMesh::calculateGlobalEdgeIndices(this->mesh->elements);
    auto const edges = std::get<0>(globalEdgeIndex);

    // fill connectivity and elementalJacobianMatrix
    auto const elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        auto const localEdges = std::get<1>(globalEdgeIndex)[element];

        // extract coordinats of node points of element
        auto const points = this->mesh->elementNodes(element);

        // fill matrix
        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            // evaluate integral equations
            Eigen::ArrayXi edgeI(2), edgeJ(2);
            edgeI << std::get<0>(std::get<1>(localEdges[i])), std::get<1>(std::get<1>(localEdges[i]));
            edgeJ << std::get<0>(std::get<1>(localEdges[j])), std::get<1>(std::get<1>(localEdges[j]));
            auto const basisI = std::make_shared<FEM::basis::Edge>(points, edgeI);
            auto const basisJ = std::make_shared<FEM::basis::Edge>(points, edgeJ);

            // set elementalJacobianMatrix element
            (*elementalJacobianMatrix)(element, i + j * 3) =
                basisI->integrateWithBasis(basisJ);
        }
    }

    elementalJacobianMatrix->copyToDevice(stream);
    this->elementalJacobianMatrix->copy(elementalJacobianMatrix, stream);
}

// calc jacobian
template <
    class dataType
>
void mpFlow::MWI::Equation<dataType>::calcJacobian(
    std::shared_ptr<numeric::Matrix<dataType> const> const field,
    cudaStream_t const stream, std::shared_ptr<numeric::Matrix<dataType>> const jacobian) const {
    // check input
    if (field == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::ellipticalEquation::calcJacobian: phi == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::ellipticalEquation::calcJacobian: jacobian == nullptr");
    }

    // dimension
    dim3 blocks(jacobian->dataRows / numeric::matrix::blockSize,
        jacobian->dataCols / numeric::matrix::blockSize);
    dim3 threads(numeric::matrix::blockSize, numeric::matrix::blockSize);

    // calc jacobian
    MWI::equationKernel::calcJacobian<dataType>(blocks, threads, stream,
        field->deviceData, this->edges->deviceData, this->elementalJacobianMatrix->deviceData,
        jacobian->dataRows, jacobian->dataCols, field->dataRows, this->mesh->elements.rows(),
        field->cols, jacobian->deviceData);

}

template <
    class dataType
>
void mpFlow::MWI::Equation<dataType>::update(std::shared_ptr<numeric::Matrix<dataType> const> const beta,
    dataType const k, cudaStream_t const stream) {
    // check input
    if (beta == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Equation::update: beta == nullptr");
    }

    // update matrices
    auto const alpha = std::make_shared<numeric::Matrix<dataType>>(beta->rows, beta->cols, stream,
        dataType(1));
    FEM::equation::updateMatrix<dataType, false>(this->elementalSMatrix, alpha,
        this->connectivityMatrix, dataType(1), stream, this->sMatrix);
    FEM::equation::updateMatrix<dataType, false>(this->elementalRMatrix, beta,
        this->connectivityMatrix, dataType(1), stream, this->rMatrix);

    // update system matrix
    FEM::equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::blockSize,
        numeric::matrix::blockSize, stream, this->sMatrix->deviceValues, this->rMatrix->deviceValues,
        this->sMatrix->deviceColumnIds, this->sMatrix->density, k, this->systemMatrix->deviceValues);
}

// specializations
template class mpFlow::MWI::Equation<float>;
template class mpFlow::MWI::Equation<double>;
template class mpFlow::MWI::Equation<thrust::complex<float>>;
template class mpFlow::MWI::Equation<thrust::complex<double>>;