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
    class basisFunctionType
>
mpFlow::FEM::Equation<dataType, basisFunctionType>::Equation(
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
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements->rows, math::square(basisFunctionType::pointsPerElement),
        stream, 0.0, false);
    this->excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->nodes->rows, this->boundaryDescriptor->count, stream,
        0.0, false);

    this->initElementalMatrices(stream);
    this->initExcitationMatrix(stream);
    this->initJacobianCalculationMatrix(stream);

    // update equation
    auto alpha = std::make_shared<numeric::Matrix<dataType>>(this->mesh->elements->rows,
        1, stream, 0.0, false);
    this->update(alpha, 0.0, alpha, stream);
}

// init elemental matrices
template <
    class dataType,
    class basisFunctionType
>
void mpFlow::FEM::Equation<dataType, basisFunctionType>::initElementalMatrices(
    cudaStream_t stream) {
    // create intermediate matrices
    auto elementCount = std::make_shared<numeric::SparseMatrix<dtype::index>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    std::vector<std::shared_ptr<numeric::SparseMatrix<dtype::index>>> connectivityMatrices;
    std::vector<std::shared_ptr<numeric::SparseMatrix<dataType>>> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        // get nodes points of element
        auto nodes = mesh->elementNodes(element);

        // extract coordinats of node points of element
        std::array<std::tuple<dtype::real, dtype::real>,
            basisFunctionType::pointsPerElement> points;
        for (dtype::index i = 0; i < points.size(); ++i) {
            points[i] = std::get<1>(nodes[i]);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; i++)
        for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            size_t level = elementCount->getValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::make_shared<numeric::SparseMatrix<dtype::index>>(
                    this->mesh->nodes->rows, this->mesh->nodes->rows, stream));
                elementalSMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->mesh->nodes->rows, this->mesh->nodes->rows, stream));
                elementalRMatrices.push_back(std::make_shared<numeric::SparseMatrix<dataType>>(
                    this->mesh->nodes->rows, this->mesh->nodes->rows, stream));
            }

            // set connectivity element
            connectivityMatrices[level]->setValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]),
                element);

            // create basis functions
            auto basisI = std::make_shared<basisFunctionType>(points, i);
            auto basisJ = std::make_shared<basisFunctionType>(points, j);

            // set elemental system element
            elementalSMatrices[level]->setValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]),
                basisI->integrateGradientWithBasis(basisJ));

            // set elemental residual element
            elementalRMatrices[level]->setValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]),
                basisI->integrateWithBasis(basisJ));

            // increment element count
            elementCount->setValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]),
                elementCount->getValue(std::get<0>(nodes[i]), std::get<0>(nodes[j])) + 1);
        }
    }

    // determine nodes with common element
    auto commonElementMatrix = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        auto nodes = this->mesh->elementNodes(element);

        for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; ++i)
        for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; ++j) {
            commonElementMatrix->setValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]), 1.0f);
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // fill sparse matrices initially with commonElementMatrix
    this->sMatrix->copy(commonElementMatrix, stream);
    this->rMatrix->copy(commonElementMatrix, stream);
    this->systemMatrix->copy(commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh->nodes->rows, numeric::sparseMatrix::block_size * connectivityMatrices.size(),
        stream, dtype::invalid_index);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * elementalSMatrices.size(), stream);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (dtype::index level = 0; level < connectivityMatrices.size(); ++level) {
        for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
            // get element nodes
            auto nodes = this->mesh->elementNodes(element);

            for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; ++i)
            for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; ++j) {
                dtype::index columId = commonElementMatrix->getColumnId(std::get<0>(nodes[i]),
                    std::get<0>(nodes[j]));

                (*this->connectivityMatrix)(std::get<0>(nodes[i]), level * numeric::sparseMatrix::block_size + columId) =
                    connectivityMatrices[level]->getValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]));
                (*this->elementalSMatrix)(std::get<0>(nodes[i]), level * numeric::sparseMatrix::block_size + columId) =
                    elementalSMatrices[level]->getValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]));
                (*this->elementalRMatrix)(std::get<0>(nodes[i]), level * numeric::sparseMatrix::block_size + columId) =
                    elementalRMatrices[level]->getValue(std::get<0>(nodes[i]), std::get<0>(nodes[j]));
            }
        }
    }
    this->connectivityMatrix->copyToDevice(stream);
    this->elementalSMatrix->copyToDevice(stream);
    this->elementalRMatrix->copyToDevice(stream);
}

template <
    class dataType,
    class basisFunctionType
>
void mpFlow::FEM::Equation<dataType, basisFunctionType>::initExcitationMatrix(cudaStream_t stream) {
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
    std::array<dtype::real, basisFunctionType::pointsPerEdge> nodeParameter;
    dtype::real integrationStart, integrationEnd;

    // calc excitation matrix
    auto excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->excitationMatrix->rows, this->excitationMatrix->cols, stream);
    for (dtype::index boundaryElement = 0; boundaryElement < this->mesh->boundary->rows; ++boundaryElement) {
        // get boundary nodes
        nodes = this->mesh->boundaryNodes(boundaryElement);

        // sort nodes by parameter
        std::sort(nodes.begin(), nodes.end(),
            [](const std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>& a,
                const std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>& b)
                -> bool {
                    return math::circleParameter(std::get<1>(b),
                        math::circleParameter(std::get<1>(a), 0.0)) > 0.0;
        });

        // calc parameter offset
        dtype::real parameterOffset = math::circleParameter(std::get<1>(nodes[0]), 0.0);

        // calc node parameter centered to node 0
        for (dtype::size i = 0; i < nodes.size(); ++i) {
            nodeParameter[i] = math::circleParameter(std::get<1>(nodes[i]),
                parameterOffset);
        }

        for (dtype::index piece = 0; piece < this->boundaryDescriptor->count; ++piece) {
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
                for (dtype::index node = 0; node < basisFunctionType::pointsPerEdge; ++node) {
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
    class basisFunctionType
>
void mpFlow::FEM::Equation<dataType, basisFunctionType>
    ::initJacobianCalculationMatrix(cudaStream_t stream) {
    // variables
    std::array<std::tuple<dtype::real, dtype::real>,
       basisFunctionType::pointsPerElement> nodeCoordinates;
    std::array<std::shared_ptr<basisFunctionType>,
        basisFunctionType::pointsPerElement> basisFunction;

    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        // get element nodes
        auto nodes = this->mesh->elementNodes(element);

        // extract nodes coordinates
        for (dtype::index node = 0; node < basisFunctionType::pointsPerElement; ++node) {
            nodeCoordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < basisFunctionType::pointsPerElement; ++node) {
            basisFunction[node] = std::make_shared<basisFunctionType>(
                nodeCoordinates, node);
        }

        // fill matrix
        for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; ++i)
        for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; ++j) {
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
    class basisFunctionType
>
void mpFlow::FEM::Equation<dataType, basisFunctionType>::update(
    const std::shared_ptr<numeric::Matrix<dataType>> alpha, const dataType k,
    const std::shared_ptr<numeric::Matrix<dataType>> beta, cudaStream_t stream) {
    // update matrices
    FEM::equation::updateMatrix(this->elementalSMatrix, alpha, this->connectivityMatrix,
        this->referenceValue, stream, this->sMatrix);
    FEM::equation::updateMatrix(this->elementalRMatrix, beta, this->connectivityMatrix,
        this->referenceValue, stream, this->rMatrix);

    // update system matrix
    FEM::equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix->deviceValues, this->rMatrix->deviceValues,
        this->sMatrix->deviceColumnIds, this->sMatrix->density, k, this->systemMatrix->deviceValues);
}

// calc jacobian
template <
    class dataType,
    class basisFunctionType
>
void mpFlow::FEM::Equation<dataType, basisFunctionType>::calcJacobian(
    const std::shared_ptr<numeric::Matrix<dataType>> phi,
    const std::shared_ptr<numeric::Matrix<dataType>> gamma,
    dtype::size driveCount, dtype::size measurmentCount, bool additiv,
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
    FEM::equationKernel::calcJacobian<basisFunctionType::pointsPerElement>(blocks, threads, stream,
        phi->deviceData, &phi->deviceData[driveCount * phi->dataRows],
        this->mesh->elements->deviceData, this->elementalJacobianMatrix->deviceData,
        gamma->deviceData, this->referenceValue, jacobian->dataRows, jacobian->dataCols,
        phi->dataRows, this->mesh->elements->rows, driveCount, measurmentCount, additiv,
        jacobian->deviceData);
}

// update matrix
template <
    class dataType
>
void mpFlow::FEM::equation::updateMatrix(
    const std::shared_ptr<numeric::Matrix<dataType>> elements,
    const std::shared_ptr<numeric::Matrix<dataType>> gamma,
    const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix,
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
    FEM::equationKernel::updateMatrix(blocks, threads, stream,
        connectivityMatrix->deviceData, elements->deviceData, gamma->deviceData,
        referenceValue, connectivityMatrix->dataRows, connectivityMatrix->dataCols, matrix->deviceValues);
}

// specialisation
template void mpFlow::FEM::equation::updateMatrix<mpFlow::dtype::real>(
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>,
    mpFlow::dtype::real, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<mpFlow::dtype::real>>);
template void mpFlow::FEM::equation::updateMatrix<mpFlow::dtype::complex>(
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::complex>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::complex>>,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>,
    mpFlow::dtype::complex, cudaStream_t, std::shared_ptr<mpFlow::numeric::SparseMatrix<mpFlow::dtype::complex>>);

template class mpFlow::FEM::Equation<mpFlow::dtype::real, mpFlow::FEM::basis::Linear>;
template class mpFlow::FEM::Equation<mpFlow::dtype::real, mpFlow::FEM::basis::Quadratic>;
