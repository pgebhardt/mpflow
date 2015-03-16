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

template <
    class dataType,
    class basisFunctionType,
    bool logarithmic
>
mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::Equation(
    std::shared_ptr<numeric::IrregularMesh> mesh,
    std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor,
    dataType referenceValue, cudaStream_t stream)
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
    cudaStream_t stream) {
    // create intermediate matrices
    auto elementCount = std::make_shared<numeric::SparseMatrix<unsigned>>(
        this->mesh->nodes.rows(), this->mesh->nodes.rows(), stream);
    std::vector<std::shared_ptr<numeric::SparseMatrix<unsigned>>> connectivityMatrices;
    std::vector<std::shared_ptr<numeric::SparseMatrix<dataType>>> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (unsigned element = 0; element < this->mesh->elements.rows(); ++element) {
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
    for (unsigned element = 0; element < this->mesh->elements.rows(); ++element) {
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
        for (unsigned element = 0; element < this->mesh->elements.rows(); ++element) {
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
void mpFlow::FEM::Equation<dataType, basisFunctionType, logarithmic>::initExcitationMatrix(cudaStream_t stream) {
    std::vector<std::tuple<unsigned, std::tuple<double, double>>> nodes;
    Eigen::ArrayXd nodeParameter(basisFunctionType::pointsPerEdge);
    double integrationStart, integrationEnd;

    // calc excitation matrix
    auto excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->excitationMatrix->rows, this->excitationMatrix->cols, stream);
    for (unsigned boundaryElement = 0; boundaryElement < this->mesh->boundary.rows(); ++boundaryElement) {
        // get boundary nodes
        nodes = this->mesh->boundaryNodes(boundaryElement);

        // sort nodes by parameter
        std::sort(nodes.begin(), nodes.end(),
            [](const std::tuple<unsigned, std::tuple<double, double>>& a,
                const std::tuple<unsigned, std::tuple<double, double>>& b)
                -> bool {
                    return math::circleParameter(std::get<1>(b),
                        math::circleParameter(std::get<1>(a), 0.0)) > 0.0;
        });

        // calc parameter offset
        double parameterOffset = math::circleParameter(std::get<1>(nodes[0]), 0.0);

        // calc node parameter centered to node 0
        for (unsigned i = 0; i < nodes.size(); ++i) {
            nodeParameter(i) = math::circleParameter(std::get<1>(nodes[i]),
                parameterOffset);
        }

        for (unsigned piece = 0; piece < this->boundaryDescriptor->count; ++piece) {
            // skip boundary part, if radii dont match
            if (std::abs(
                std::get<0>(math::polar(std::get<0>(this->boundaryDescriptor->coordinates[piece]))) -
                std::get<0>(math::polar(std::get<1>(nodes[0])))) > 0.0001) {
                continue;
            }

            // calc integration interval centered to node 0
            integrationStart = math::circleParameter(
                std::get<0>(this->boundaryDescriptor->coordinates[piece]),
                parameterOffset);
            integrationEnd = math::circleParameter(
                std::get<1>(this->boundaryDescriptor->coordinates[piece]),
                parameterOffset);

            // intgrate if integrationStart is left of integrationEnd
            if (integrationStart < integrationEnd) {
                // calc element
                for (unsigned node = 0; node < basisFunctionType::pointsPerEdge; ++node) {
                    (*excitationMatrix)(std::get<0>(nodes[node]), piece) +=
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
    ::initJacobianCalculationMatrix(cudaStream_t stream) {
    // variables
    std::array<std::shared_ptr<basisFunctionType>,
        basisFunctionType::pointsPerElement> basisFunction;

    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (unsigned element = 0; element < this->mesh->elements.rows(); ++element) {
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
    const std::shared_ptr<numeric::Matrix<dataType>> alpha, const dataType k,
    const std::shared_ptr<numeric::Matrix<dataType>> beta, cudaStream_t stream) {
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
    const std::shared_ptr<numeric::Matrix<dataType>> phi,
    const std::shared_ptr<numeric::Matrix<dataType>> gamma,
    unsigned driveCount, unsigned measurmentCount, bool additiv,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<dataType>> jacobian) {
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
    const std::shared_ptr<numeric::Matrix<dataType>> elements,
    const std::shared_ptr<numeric::Matrix<dataType>> gamma,
    const std::shared_ptr<numeric::Matrix<unsigned>> connectivityMatrix,
    dataType referenceValue, cudaStream_t stream, std::shared_ptr<numeric::SparseMatrix<dataType>> matrix) {
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
    const std::shared_ptr<mpFlow::numeric::Matrix<float>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<float>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    float, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>);
template void mpFlow::FEM::equation::updateMatrix<float, false>(
    const std::shared_ptr<mpFlow::numeric::Matrix<float>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<float>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    float, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<float>>);
template void mpFlow::FEM::equation::updateMatrix<double, true>(
    const std::shared_ptr<mpFlow::numeric::Matrix<double>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<double>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    double, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>);
template void mpFlow::FEM::equation::updateMatrix<double, false>(
    const std::shared_ptr<mpFlow::numeric::Matrix<double>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<double>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    double, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<double>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<float>, true>(
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    thrust::complex<float>, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<float>, false>(
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    thrust::complex<float>, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<double>, true>(
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    thrust::complex<double>, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>);
template void mpFlow::FEM::equation::updateMatrix<thrust::complex<double>, false>(
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<unsigned>>,
    thrust::complex<double>, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>>);

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
