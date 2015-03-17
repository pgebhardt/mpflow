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
#include <iostream>

template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::Equation(
    std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<FEM::BoundaryDescriptor const> const boundaryDescriptor,
    dataType const referenceValue, cudaStream_t const stream)
    : mesh(mesh), boundaryDescriptor(boundaryDescriptor), referenceValue(referenceValue) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::Equation::Equation: mesh == nullptr");
    }
    if (boundaryDescriptor == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::Equation::Equation: boundaryDescriptor == nullptr");
    }

    // init matrices
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream);
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements.rows(), math::square(basisFunctionType::pointsPerElement),
        stream, 0.0, false);
    this->excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->nodes.rows(), this->boundaryDescriptor->count, stream,
        0.0, false);
    this->meshElements = numeric::matrix::fromEigen<unsigned, int>(this->mesh->elements, stream);

    this->initElementalMatrices(stream);
    this->initExcitationMatrix(stream);
    this->initJacobianCalculationMatrix(stream);

    // update equation
    auto alpha = std::make_shared<numeric::Matrix<dataType>>(this->mesh->elements.rows(),
        1, stream, 0.0, false);
    this->update(alpha, 0.0, alpha, stream);
}

// init elemental matrices
template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
void mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::initElementalMatrices(
    cudaStream_t const stream) {
    // create intermediate matrices
    auto elementCount = std::make_shared<numeric::SparseMatrix<unsigned>>(
        this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream);
    std::vector<std::shared_ptr<numeric::SparseMatrix<unsigned>>> connectivityMatrices;
    std::vector<std::shared_ptr<numeric::SparseMatrix<dataType>>> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        // get nodes points of element
        Eigen::ArrayXXd points = mesh->elementNodes(element);

        // set connectivity and elemental residual matrix elements
        for (unsigned i = 0; i < basisFunctionType::pointsPerElement; i++)
        for (unsigned j = 0; j < basisFunctionType::pointsPerElement; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            size_t level = elementCount->getValue(this->mesh->elements(element, i), this->mesh->elements(element, j));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::make_shared<numeric::SparseMatrix<unsigned>>(
                    this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream));
                elementalSMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream));
                elementalRMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream));
            }

            // set connectivity element
            connectivityMatrices[level]->setValue(this->mesh->elements(element, i),
                this->mesh->elements(element, j), element);

            // create basis functions
            auto basisI = std::make_shared<basisFunctionType>(points, i);
            auto basisJ = std::make_shared<basisFunctionType>(points, j);

            // set elemental system element
            elementalSMatrices[level]->setValue(this->mesh->elements(element, i),
                this->mesh->elements(element, j), basisI->integrateGradientWithBasis(basisJ));

            // set elemental residual element
            elementalRMatrices[level]->setValue(this->mesh->elements(element, i),
                this->mesh->elements(element, j), basisI->integrateWithBasis(basisJ));

            // increment element count
            elementCount->setValue(this->mesh->elements(element, i), this->mesh->elements(element, j),
                elementCount->getValue(this->mesh->elements(element, i), this->mesh->elements(element, j)) + 1);
        }
    }

    // determine nodes with common element
    auto commonElementMatrix = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        for (unsigned i = 0; i < basisFunctionType::pointsPerElement; ++i)
        for (unsigned j = 0; j < basisFunctionType::pointsPerElement; ++j) {
            commonElementMatrix->setValue(this->mesh->elements(element, i), this->mesh->elements(element, j), 1.0f);
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // fill sparse matrices initially with commonElementMatrix
    this->sMatrix->copy(commonElementMatrix, stream);
    this->rMatrix->copy(commonElementMatrix, stream);
    this->systemMatrix->copy(commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<unsigned>>(
        this->mesh->nodes.rows(), numeric::sparseMatrix::block_size * connectivityMatrices.size(),
        stream, constants::invalid_index);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes.rows(),
        numeric::sparseMatrix::block_size * elementalSMatrices.size(), stream);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes.rows(),
        numeric::sparseMatrix::block_size * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (unsigned level = 0; level < connectivityMatrices.size(); ++level) {
        for (int element = 0; element < this->mesh->elements.rows(); ++element) {
            for (unsigned i = 0; i < basisFunctionType::pointsPerElement; ++i)
            for (unsigned j = 0; j < basisFunctionType::pointsPerElement; ++j) {
                unsigned columId = commonElementMatrix->getColumnId(this->mesh->elements(element, i),
                    this->mesh->elements(element, j));

                (*this->connectivityMatrix)(this->mesh->elements(element, i), level * numeric::sparseMatrix::block_size + columId) =
                    connectivityMatrices[level]->getValue(this->mesh->elements(element, i), this->mesh->elements(element, j));
                (*this->elementalSMatrix)(this->mesh->elements(element, i), level * numeric::sparseMatrix::block_size + columId) =
                    elementalSMatrices[level]->getValue(this->mesh->elements(element, i), this->mesh->elements(element, j));
                (*this->elementalRMatrix)(this->mesh->elements(element, i), level * numeric::sparseMatrix::block_size + columId) =
                    elementalRMatrices[level]->getValue(this->mesh->elements(element, i), this->mesh->elements(element, j));
            }
        }
    }
    this->connectivityMatrix->copyToDevice(stream);
    this->elementalSMatrix->copyToDevice(stream);
    this->elementalRMatrix->copyToDevice(stream);
}

template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
void mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::initExcitationMatrix(cudaStream_t const stream) {
    Eigen::ArrayXd nodeParameter(basisFunctionType::pointsPerEdge);
    double integrationStart, integrationEnd;

    // calc excitation matrix
    auto excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->excitationMatrix->rows, this->excitationMatrix->cols, stream);
    for (int boundaryElement = 0; boundaryElement < this->mesh->boundary.rows(); ++boundaryElement) {
        // get boundary nodes
        Eigen::ArrayXXd nodes = this->mesh->boundaryNodes(boundaryElement);

        // sort nodes by parameter
        std::vector<Eigen::ArrayXd> nodesVector(nodes.rows());
        for (unsigned i = 0; i < nodes.rows(); ++i) {
            nodesVector[i] = nodes.row(i);
        }
        std::sort(nodesVector.begin(), nodesVector.end(),
            [](Eigen::Ref<const Eigen::ArrayXd> a, Eigen::Ref<const Eigen::ArrayXd> b) -> bool {
            return math::circleParameter(b, math::circleParameter(a, 0.0)) > 0.0;
        });
        for (unsigned i = 0; i < nodes.rows(); ++i) {
            nodes.row(i) = nodesVector[i];
        }

        // calc parameter offset
        double parameterOffset = math::circleParameter(nodes.row(0).transpose(), 0.0);

        // calc node parameter centered to node 0
        for (unsigned i = 0; i < nodes.rows(); ++i) {
            nodeParameter(i) = math::circleParameter(nodes.row(i).transpose(), parameterOffset);
        }

        for (unsigned piece = 0; piece < this->boundaryDescriptor->count; ++piece) {
            // skip boundary part, if radii dont match
            if (std::abs(math::polar(this->boundaryDescriptor->coordinates.block(piece, 0, 1, 2).transpose())(0) -
                math::polar(nodes.row(0).transpose())(0)) > 1e-4) {
                continue;
            }

            // calc integration interval centered to node 0
            integrationStart = math::circleParameter(this->boundaryDescriptor->coordinates.block(piece, 0, 1, 2).transpose(),
                parameterOffset);
            integrationEnd = math::circleParameter(this->boundaryDescriptor->coordinates.block(piece, 2, 1, 2).transpose(),
                parameterOffset);

            // intgrate if integrationStart is left of integrationEnd
            if (integrationStart < integrationEnd) {
                // calc element
                for (unsigned node = 0; node < basisFunctionType::pointsPerEdge; ++node) {
                    (*excitationMatrix)(this->mesh->boundary(boundaryElement, node), piece) +=
                        basisFunctionType::integrateBoundaryEdge(
                            nodeParameter, node, integrationStart, integrationEnd) /
                        std::get<0>(this->boundaryDescriptor->shapes[piece]);
                }
            }
        }
    }

    excitationMatrix->copyToDevice(stream);
    this->excitationMatrix->copy(excitationMatrix, stream);
}

template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
void mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>
    ::initJacobianCalculationMatrix(cudaStream_t const stream) {
    // variables
    std::array<std::shared_ptr<basisFunctionType>,
        basisFunctionType::pointsPerElement> basisFunction;

    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (int element = 0; element < this->mesh->elements.rows(); ++element) {
        // get element points
        Eigen::ArrayXXd points = this->mesh->elementNodes(element);

        // calc corresponding basis functions
        for (unsigned node = 0; node < basisFunctionType::pointsPerElement; ++node) {
            basisFunction[node] = std::make_shared<basisFunctionType>(
                points, node);
        }

        // fill matrix
        for (unsigned i = 0; i < basisFunctionType::pointsPerElement; ++i)
        for (unsigned j = 0; j < basisFunctionType::pointsPerElement; ++j) {
            // set elementalJacobianMatrix element
            (*elementalJacobianMatrix)(element, i +
                j * basisFunctionType::pointsPerElement) =
                basisFunction[i]->integrateGradientWithBasis(basisFunction[j]);
        }
    }

    elementalJacobianMatrix->copyToDevice(stream);
    this->elementalJacobianMatrix->copy(elementalJacobianMatrix, stream);
}

// update ellipticalEquation
template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
void mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::update(
    std::shared_ptr<numeric::Matrix<dataType> const> const alpha, dataType const k,
    std::shared_ptr<numeric::Matrix<dataType> const> const beta, cudaStream_t const stream) {
    // update matrices
    FEM::equation::updateMatrix<dataType, logarithmic>(this->elementalSMatrix, alpha,
        this->connectivityMatrix, this->referenceValue, stream, this->sMatrix);
    FEM::equation::updateMatrix<dataType, logarithmic>(this->elementalRMatrix, beta,
        this->connectivityMatrix, this->referenceValue, stream, this->rMatrix);

    // update system matrix
    FEM::equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix->deviceValues, this->rMatrix->deviceValues,
        this->sMatrix->deviceColumnIds, this->sMatrix->density, k, this->systemMatrix->deviceValues);
}

// calc jacobian
template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
void mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::calcJacobian(
    std::shared_ptr<numeric::Matrix<dataType> const> const phi,
    std::shared_ptr<numeric::Matrix<dataType> const> const gamma,
    unsigned const driveCount, unsigned const measurmentCount, bool const additiv,
    cudaStream_t const stream, std::shared_ptr<numeric::Matrix<dataType>> jacobian) const {
    // check input
    if (phi == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::ellipticalEquation::calcJacobian: phi == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::ellipticalEquation::calcJacobian: gamma == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::ellipticalEquation::calcJacobian: jacobian == nullptr");
    }

    // dimension
    dim3 blocks(jacobian->dataRows / numeric::matrix::block_size,
        jacobian->dataCols / numeric::matrix::block_size);
    dim3 threads(numeric::matrix::block_size, numeric::matrix::block_size);

    // calc jacobian
    FEM::equationKernel::calcJacobian<dataType, basisFunctionType::pointsPerElement, logarithmic>(
        blocks, threads, stream, phi->deviceData, &phi->deviceData[driveCount * phi->dataRows],
        this->meshElements->deviceData, this->elementalJacobianMatrix->deviceData,
        gamma->deviceData, this->referenceValue, jacobian->dataRows, jacobian->dataCols,
        phi->dataRows, this->meshElements->rows, driveCount, measurmentCount, additiv,
        jacobian->deviceData);
}

// update matrix
template <
    class dataType,
    bool logarithmic
>
void mpFlow::FEM::equation::updateMatrix(
    std::shared_ptr<numeric::Matrix<dataType> const> const elements,
    std::shared_ptr<numeric::Matrix<dataType> const> const gamma,
    std::shared_ptr<numeric::Matrix<unsigned> const> const connectivityMatrix,
    dataType const referenceValue, cudaStream_t const stream,
    std::shared_ptr<numeric::SparseMatrix<dataType>> matrix) {
    // check input
    if (elements == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::updateMatrix: elements == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::updateMatrix: gamma == nullptr");
    }
    if (connectivityMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::updateMatrix: connectivityMatrix == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::updateMatrix: matrix == nullptr");
    }

    // dimension
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);
    dim3 blocks(matrix->dataRows / numeric::matrix::block_size, 1);

    // execute kernel
    FEM::equationKernel::updateMatrix<dataType, logarithmic>(blocks, threads, stream,
        connectivityMatrix->deviceData, elements->deviceData, gamma->deviceData,
        referenceValue, connectivityMatrix->dataRows, connectivityMatrix->dataCols, matrix->deviceValues);
}

// specialisation
template void mpFlow::FEM::equation::updateMatrix<float, true>(
    std::shared_ptr<mpFlow::numeric::Matrix<float> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<float> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    float const, cudaStream_t const, std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>);
template void mpFlow::FEM::equation::updateMatrix<float, false>(
    std::shared_ptr<mpFlow::numeric::Matrix<float> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<float> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    float const, cudaStream_t const, std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>);
template void mpFlow::FEM::equation::updateMatrix<double, true>(
    std::shared_ptr<mpFlow::numeric::Matrix<double> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<double> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    double const, cudaStream_t const, std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>);
template void mpFlow::FEM::equation::updateMatrix<double, false>(
    std::shared_ptr<mpFlow::numeric::Matrix<double> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<double> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    double const, cudaStream_t const, std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<float>, true>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    thrust::complex<float> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<float>, false>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    thrust::complex<float> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<double>, true>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    thrust::complex<double> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<double>, false>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>> const> const,
    std::shared_ptr<mpFlow::numeric::Matrix<unsigned> const> const,
    thrust::complex<double> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>);

template class mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>;
template class mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Quadratic, true>;
template class mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>;
template class mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Quadratic, false>;
template class mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>;
template class mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Quadratic, true>;
template class mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>;
template class mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Quadratic, false>;
template class mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>;
template class mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Quadratic, true>;
template class mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>;
template class mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Quadratic, false>;
template class mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>;
template class mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Quadratic, true>;
template class mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>;
template class mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Quadratic, false>;
