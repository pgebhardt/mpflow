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
#include "mpflow/eit/equation_kernel.h"

template <
    class basisFunctionType
>
mpFlow::EIT::Equation<basisFunctionType>::Equation(
    std::shared_ptr<numeric::IrregularMesh> mesh,
    std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor,
    dtype::real referenceValue, cudaStream_t stream)
    : mesh(mesh), boundaryDescriptor(boundaryDescriptor), referenceValue(referenceValue) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Equation::Equation: mesh == nullptr");
    }
    if (boundaryDescriptor == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Equation::Equation: boundaryDescriptor == nullptr");
    }

    // init matrices
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh->elements->rows,
        math::square(basisFunctionType::nodesPerElement), stream);
    this->excitationMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh->nodes->rows, this->boundaryDescriptor->count, stream);

    auto commonElementMatrix = this->initElementalMatrices(stream);
    this->initExcitationMatrix(stream);
    this->initJacobianCalculationMatrix(stream);

    // create initial sparse system matrix from common element Matrix
    this->systemMatrix = std::make_shared<numeric::SparseMatrix<dtype::real>>(
        commonElementMatrix, stream);

    // update ellipticalEquation
    auto gamma = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh->elements->rows, 1, stream);
    this->update(gamma, 0.0, stream);
}

// init elemental matrices
template <
    class basisFunctionType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::Equation<basisFunctionType>::initElementalMatrices(
    cudaStream_t stream) {
    // create intermediate matrices
    std::vector<std::vector<dtype::index>> elementCount(
        this->mesh->nodes->rows, std::vector<dtype::index>(
        this->mesh->nodes->rows, 0));
    std::vector<std::vector<std::vector<dtype::index>>> connectivityMatrices;
    std::vector<std::vector<std::vector<dtype::real>>> elementalSMatrices,
        elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
    std::array<std::tuple<dtype::real, dtype::real>,
        basisFunctionType::nodesPerElement> nodeCoordinates;
    std::array<std::shared_ptr<basisFunctionType>,
        basisFunctionType::nodesPerElement> basisFunction;
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        // get element nodes
        nodes = this->mesh->elementNodes(element);

        // extract coordinates
        for (dtype::index node = 0; node < basisFunctionType::nodesPerElement; ++node) {
            nodeCoordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < basisFunctionType::nodesPerElement; ++node) {
            basisFunction[node] = std::make_shared<basisFunctionType>(
                nodeCoordinates, node);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < basisFunctionType::nodesPerElement; i++)
        for (dtype::index j = 0; j < basisFunctionType::nodesPerElement; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            auto level = elementCount[std::get<0>(nodes[i])][std::get<0>(nodes[j])];
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::vector<std::vector<dtype::index>>(
                    this->mesh->nodes->rows, std::vector<dtype::index>(
                    this->mesh->nodes->rows, dtype::invalid_index)));
                elementalSMatrices.push_back(std::vector<std::vector<dtype::real>>(
                    this->mesh->nodes->rows, std::vector<dtype::real>(
                    this->mesh->nodes->rows, 0.0)));
                elementalRMatrices.push_back(std::vector<std::vector<dtype::real>>(
                    this->mesh->nodes->rows, std::vector<dtype::real>(
                    this->mesh->nodes->rows, 0.0)));
            }

            // set connectivity element
            connectivityMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                element;

            // set elemental system element
            elementalSMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                basisFunction[i]->integrateGradientWithBasis(basisFunction[j]);

            // set elemental residual element
            elementalRMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                basisFunction[i]->integrateWithBasis(basisFunction[j]);

            // increment element count
            elementCount[std::get<0>(nodes[i])][std::get<0>(nodes[j])]++;
        }
    }

    // determine nodes with common element
    auto commonElementMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        nodes = this->mesh->elementNodes(element);

        for (dtype::index i = 0; i < basisFunctionType::nodesPerElement; ++i)
        for (dtype::index j = 0; j < basisFunctionType::nodesPerElement; ++j) {
            (*commonElementMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) = 1.0f;
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // create sparse matrices
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        commonElementMatrix, stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * connectivityMatrices.size(), stream, dtype::invalid_index);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * elementalSMatrices.size(), stream);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    auto connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream,
        dtype::invalid_index);
    auto elementalSMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    auto elementalRMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    for (dtype::index level = 0; level < connectivityMatrices.size(); ++level) {
        for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
            // get element nodes
            nodes = this->mesh->elementNodes(element);

            for (dtype::index i = 0; i < basisFunctionType::nodesPerElement; ++i)
            for (dtype::index j = 0; j < basisFunctionType::nodesPerElement; ++j) {
                (*connectivityMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    connectivityMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
                (*elementalSMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    elementalSMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
                (*elementalRMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    elementalRMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
            }
        }
        connectivityMatrix->copyToDevice(stream);
        elementalSMatrix->copyToDevice(stream);
        elementalRMatrix->copyToDevice(stream);
        cudaStreamSynchronize(stream);

        reduceMatrix(connectivityMatrix, this->sMatrix, level, stream,
            this->connectivityMatrix);
        reduceMatrix(elementalSMatrix, this->sMatrix, level, stream,
            this->elementalSMatrix);
        reduceMatrix(elementalRMatrix, this->rMatrix, level, stream,
            this->elementalRMatrix);
    }

    return commonElementMatrix;
}

template <
    class basisFunctionType
>
void mpFlow::EIT::Equation<basisFunctionType>::initExcitationMatrix(cudaStream_t stream) {
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
    std::array<dtype::real, basisFunctionType::nodesPerEdge> nodeParameter;
    dtype::real integrationStart, integrationEnd;

    // calc excitation matrix
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
                for (dtype::index node = 0; node < basisFunctionType::nodesPerEdge; ++node) {
                    (*this->excitationMatrix)(std::get<0>(nodes[node]), piece) +=
                        basisFunctionType::integrateBoundaryEdge(
                            nodeParameter, node, integrationStart, integrationEnd) /
                        std::get<0>(this->boundaryDescriptor->shape);
                }
            }
        }
    }
    this->excitationMatrix->copyToDevice(stream);
}

template <
    class basisFunctionType
>
void mpFlow::EIT::Equation<basisFunctionType>
    ::initJacobianCalculationMatrix(cudaStream_t stream) {
    // variables
    std::array<std::tuple<dtype::real, dtype::real>,
       basisFunctionType::nodesPerElement> nodeCoordinates;
    std::array<std::shared_ptr<basisFunctionType>,
        basisFunctionType::nodesPerElement> basisFunction;

    // fill connectivity and elementalJacobianMatrix
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        // get element nodes
        auto nodes = this->mesh->elementNodes(element);

        // extract nodes coordinates
        for (dtype::index node = 0; node < basisFunctionType::nodesPerElement; ++node) {
            nodeCoordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < basisFunctionType::nodesPerElement; ++node) {
            basisFunction[node] = std::make_shared<basisFunctionType>(
                nodeCoordinates, node);
        }

        // fill matrix
        for (dtype::index i = 0; i < basisFunctionType::nodesPerElement; ++i)
        for (dtype::index j = 0; j < basisFunctionType::nodesPerElement; ++j) {
            // set elementalJacobianMatrix element
            (*this->elementalJacobianMatrix)(element, i +
                j * basisFunctionType::nodesPerElement) =
                basisFunction[i]->integrateGradientWithBasis(basisFunction[j]);
        }
    }
    this->elementalJacobianMatrix->copyToDevice(stream);
}

// update ellipticalEquation
template <
    class basisFunctionType
>
void mpFlow::EIT::Equation<basisFunctionType>::update(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, dtype::real k, cudaStream_t stream) {
    // update matrices
    updateMatrix(this->elementalSMatrix, gamma, this->connectivityMatrix,
        this->referenceValue, stream, this->sMatrix);
    updateMatrix(this->elementalRMatrix, gamma, this->connectivityMatrix,
        this->referenceValue, stream, this->rMatrix);

    // update system matrix
    equationKernel::updateSystemMatrix(this->sMatrix->dataRows / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix->values, this->rMatrix->values,
        this->sMatrix->columnIds, this->sMatrix->density, k, this->systemMatrix->values);
}

// calc jacobian
template <
    class basisFunctionType
>
void mpFlow::EIT::Equation<basisFunctionType>::calcJacobian(
    const std::shared_ptr<numeric::Matrix<dtype::real>> phi,
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
    dtype::size driveCount, dtype::size measurmentCount, bool additiv,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<dtype::real>> jacobian) {
    // check input
    if (phi == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::calcJacobian: phi == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::calcJacobian: gamma == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::calcJacobian: jacobian == nullptr");
    }

    // dimension
    dim3 blocks(jacobian->dataRows / numeric::matrix::block_size,
        jacobian->dataCols / numeric::matrix::block_size);
    dim3 threads(numeric::matrix::block_size, numeric::matrix::block_size);

    // calc jacobian
    equationKernel::calcJacobian<basisFunctionType::nodesPerElement>(blocks, threads, stream,
        phi->deviceData, &phi->deviceData[driveCount * phi->dataRows],
        this->mesh->elements->deviceData, this->elementalJacobianMatrix->deviceData,
        gamma->deviceData, this->referenceValue, jacobian->dataRows, jacobian->dataCols,
        phi->dataRows, this->mesh->elements->rows, driveCount, measurmentCount, additiv,
        jacobian->deviceData);
}

// reduce matrix
template <
    class basisFunctionType
>
template <
    class type
>
void mpFlow::EIT::Equation<basisFunctionType>::reduceMatrix(
    const std::shared_ptr<numeric::Matrix<type>> intermediateMatrix,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>> shape, dtype::index offset,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<type>> matrix) {
    // check input
    if (intermediateMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::reduceMatrix: intermediateMatrix == nullptr");
    }
    if (shape == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::reduceMatrix: shape == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::reduceMatrix: matrix == nullptr");
    }

    // block size
    dim3 blocks(matrix->dataRows / numeric::matrix::block_size, 1);
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);

    // reduce matrix
    equationKernel::reduceMatrix<type>(blocks, threads, stream,
        intermediateMatrix->deviceData, shape->columnIds, matrix->dataRows,
        offset, matrix->deviceData);
}

// update matrix
template <
    class basisFunctionType
>
void mpFlow::EIT::Equation<basisFunctionType>::updateMatrix(
    const std::shared_ptr<numeric::Matrix<dtype::real>> elements,
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
    const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix,
    dtype::real sigmaRef, cudaStream_t stream, std::shared_ptr<numeric::SparseMatrix<dtype::real>> matrix) {
    // check input
    if (elements == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::updateMatrix: elements == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::updateMatrix: gamma == nullptr");
    }
    if (connectivityMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::updateMatrix: connectivityMatrix == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ellipticalEquation::updateMatrix: matrix == nullptr");
    }

    // dimension
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);
    dim3 blocks(matrix->dataRows / numeric::matrix::block_size, 1);

    // execute kernel
    equationKernel::updateMatrix(blocks, threads, stream,
        connectivityMatrix->deviceData, elements->deviceData, gamma->deviceData,
        sigmaRef, connectivityMatrix->dataRows, connectivityMatrix->dataCols, matrix->values);
}

// specialisation
template void mpFlow::EIT::Equation<mpFlow::FEM::basis::Linear>::reduceMatrix<mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::EIT::Equation<mpFlow::FEM::basis::Linear>::reduceMatrix<mpFlow::dtype::index>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);
template void mpFlow::EIT::Equation<mpFlow::FEM::basis::Quadratic>::reduceMatrix<mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::EIT::Equation<mpFlow::FEM::basis::Quadratic>::reduceMatrix<mpFlow::dtype::index>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);

template class mpFlow::EIT::Equation<mpFlow::FEM::basis::Linear>;
template class mpFlow::EIT::Equation<mpFlow::FEM::basis::Quadratic>;
