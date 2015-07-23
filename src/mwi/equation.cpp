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
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->edges.rows(), this->mesh->edges.rows(), stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->edges.rows(), this->mesh->edges.rows(), stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->edges.rows(), this->mesh->edges.rows(), stream);
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements.rows(), math::square(3), stream, 0.0, false);
    this->elementEdges = numeric::Matrix<int>::fromEigen(this->mesh->elementEdges, stream);
    
    this->initElementalMatrices(stream);
    this->initJacobianCalculationMatrix(stream);
}

Eigen::ArrayXi getLocalEdge(Eigen::Ref<Eigen::ArrayXXi const> const elements,
    unsigned const element, unsigned const localEdgeIndex) {
    Eigen::ArrayXi edge(2);

    if (elements(element, localEdgeIndex) < elements(element, (localEdgeIndex + 1) % elements.cols())) {
        edge << localEdgeIndex, (localEdgeIndex + 1) % elements.cols();
    }
    else {
        edge << (localEdgeIndex + 1) % elements.cols(), localEdgeIndex;
    }
    
    return edge;
}

// init elemental matrices
template <
    class dataType
>
void mpFlow::MWI::Equation<dataType>::initElementalMatrices(cudaStream_t const stream) {
    // create intermediate matrices
    auto const elementCount = std::make_shared<numeric::SparseMatrix<unsigned>>(
        this->mesh->edges.rows(), this->mesh->edges.rows(), stream);
    std::vector<std::shared_ptr<numeric::SparseMatrix<unsigned>>> connectivityMatrices;
    std::vector<std::shared_ptr<numeric::SparseMatrix<dataType>>> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        // extract coordinats of node points of element
        auto const points = this->mesh->elementNodes(element);

        // set connectivity and elemental residual matrix elements
        for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            unsigned const level = elementCount->getValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::make_shared<numeric::SparseMatrix<unsigned>>(
                    this->mesh->edges.rows(), this->mesh->edges.rows(), stream));
                elementalSMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->mesh->edges.rows(), this->mesh->edges.rows(), stream));
                elementalRMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->mesh->edges.rows(), this->mesh->edges.rows(), stream));
            }

            // set connectivity element
            connectivityMatrices[level]->setValue(this->mesh->elementEdges(element, i),
                this->mesh->elementEdges(element, j), element);

            // evaluate integral equations
            auto const basisI = FEM::basis::Edge(points, getLocalEdge(this->mesh->elements, element, i));
            auto const basisJ = FEM::basis::Edge(points, getLocalEdge(this->mesh->elements, element, j));

            // set elemental system element
            elementalSMatrices[level]->setValue(this->mesh->elementEdges(element, i),
                this->mesh->elementEdges(element, j), basisI.integrateGradientWithBasis(basisJ));

            // set elemental residual element
            elementalRMatrices[level]->setValue(this->mesh->elementEdges(element, i),
                this->mesh->elementEdges(element, j), basisI.integrateWithBasis(basisJ));

            // increment element count
            elementCount->setValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j),
                elementCount->getValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j)) + 1);
        }
    }

    // determine nodes with common element
    auto const commonElementMatrix = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->mesh->edges.rows(), this->mesh->edges.rows(), stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            commonElementMatrix->setValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j), dataType(1));
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // fill sparse matrices initially with commonElementMatrix
    this->sMatrix->copy(commonElementMatrix, stream);
    this->rMatrix->copy(commonElementMatrix, stream);
    this->systemMatrix->copy(commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<unsigned>>(
        this->mesh->edges.rows(), numeric::sparseMatrix::blockSize * connectivityMatrices.size(),
        stream, constants::invalidIndex);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->edges.rows(),
        numeric::sparseMatrix::blockSize * elementalSMatrices.size(), stream);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->edges.rows(),
        numeric::sparseMatrix::blockSize * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (unsigned level = 0; level < connectivityMatrices.size(); ++level) {
        for (int element = 0; element < this->mesh->elements.rows(); ++element) {
            for (unsigned i = 0; i < 3; ++i)
            for (unsigned j = 0; j < 3; ++j) {
                unsigned columId = commonElementMatrix->getColumnId(this->mesh->elementEdges(element, i),
                    this->mesh->elementEdges(element, j));

                (*this->connectivityMatrix)(this->mesh->elementEdges(element, i), level * numeric::sparseMatrix::blockSize + columId) =
                    connectivityMatrices[level]->getValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j));
                (*this->elementalSMatrix)(this->mesh->elementEdges(element, i), level * numeric::sparseMatrix::blockSize + columId) =
                    elementalSMatrices[level]->getValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j));
                (*this->elementalRMatrix)(this->mesh->elementEdges(element, i), level * numeric::sparseMatrix::blockSize + columId) =
                    elementalRMatrices[level]->getValue(this->mesh->elementEdges(element, i), this->mesh->elementEdges(element, j));
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
    // fill connectivity and elementalJacobianMatrix
    auto const elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        // extract coordinats of node points of element
        auto const points = this->mesh->elementNodes(element);

        // fill matrix
        for (unsigned i = 0; i < 3; ++i)
        for (unsigned j = 0; j < 3; ++j) {
            // evaluate integral equations
            auto const basisI = FEM::basis::Edge(points, getLocalEdge(this->mesh->elements, element, i));
            auto const basisJ = FEM::basis::Edge(points, getLocalEdge(this->mesh->elements, element, j));

            // set elementalJacobianMatrix element
            (*elementalJacobianMatrix)(element, i + j * 3) =
                basisI.integrateWithBasis(basisJ);
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
        field->deviceData, this->elementEdges->deviceData, this->elementalJacobianMatrix->deviceData,
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