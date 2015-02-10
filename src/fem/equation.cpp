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
    this->elementalJacobianMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
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
    Eigen::ArrayXXi elementCount = Eigen::ArrayXXi::Zero(this->mesh->nodes->rows,
        this->mesh->nodes->rows);
    std::vector<Eigen::ArrayXXi> connectivityMatrices;
    std::vector<Eigen::ArrayXXf> elementalSMatrices, elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        // get nodes points of element
        auto indices = std::get<0>(mesh->elementNodes(element));
        auto points = std::get<1>(mesh->elementNodes(element));

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; i++)
        for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            size_t level = elementCount(indices(i), indices(j));
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(Eigen::ArrayXXi::Ones(
                    this->mesh->nodes->rows, this->mesh->nodes->rows)
                    * dtype::invalid_index);
                elementalSMatrices.push_back(Eigen::ArrayXXf::Zero(
                    this->mesh->nodes->rows, this->mesh->nodes->rows));
                elementalRMatrices.push_back(Eigen::ArrayXXf::Zero(
                    this->mesh->nodes->rows, this->mesh->nodes->rows));
            }

            // set connectivity element
            connectivityMatrices[level](indices(i), indices(j)) =
                element;

            // create basis functions
            auto basisI = std::make_shared<basisFunctionType>(points, i);
            auto basisJ = std::make_shared<basisFunctionType>(points, j);

            // set elemental system element
            elementalSMatrices[level](indices(i), indices(j)) =
                basisI->integrateGradientWithBasis(basisJ);

            // set elemental residual element
            elementalRMatrices[level](indices(i), indices(j)) =
                basisI->integrateWithBasis(basisJ);

            // increment element count
            elementCount(indices(i), indices(j))++;
        }
    }

    // determine nodes with common element
    auto commonElementMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->nodes->rows, this->mesh->nodes->rows, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        auto indices = std::get<0>(this->mesh->elementNodes(element));

        for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; ++i)
        for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; ++j) {
            (*commonElementMatrix)(indices(i), indices(j)) = 1.0f;
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // create sparse matrices
    this->sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        commonElementMatrix, stream);
    this->rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        commonElementMatrix, stream);
    this->systemMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dataType>>(
        commonElementMatrix, stream);

    // create elemental matrices
    this->connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh->nodes->rows, numeric::sparseMatrix::block_size * connectivityMatrices.size(),
        stream, dtype::invalid_index, false);
    this->elementalSMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * elementalSMatrices.size(), stream, false);
    this->elementalRMatrix = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes->rows,
        numeric::sparseMatrix::block_size * elementalRMatrices.size(), stream, false);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (dtype::index level = 0; level < connectivityMatrices.size(); ++level) {
        // convert eigen array to mpFlow matrix and reduce to sparse format
        auto connectivityMatrix = numeric::matrix::fromEigen<dtype::index, Eigen::ArrayXXi::Scalar>(connectivityMatrices[level], stream);
        auto elementalSMatrix = numeric::matrix::fromEigen<dtype::real, dtype::real>(elementalSMatrices[level], stream);
        auto elementalRMatrix = numeric::matrix::fromEigen<dtype::real, dtype::real>(elementalRMatrices[level], stream);

        FEM::equation::reduceMatrix(connectivityMatrix, this->sMatrix, level, stream,
            this->connectivityMatrix);
        FEM::equation::reduceMatrix(elementalSMatrix, this->sMatrix, level, stream,
            this->elementalSMatrix);
        FEM::equation::reduceMatrix(elementalRMatrix, this->rMatrix, level, stream,
            this->elementalRMatrix);
        cudaStreamSynchronize(stream);
    }
}

template <
    class dataType,
    class basisFunctionType
>
void mpFlow::FEM::Equation<dataType, basisFunctionType>::initExcitationMatrix(cudaStream_t stream) {
    // calc excitation matrix
    auto excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->excitationMatrix->rows, this->excitationMatrix->cols, stream);
    for (dtype::index boundaryElement = 0; boundaryElement < this->mesh->boundary->rows; ++boundaryElement) {
        // get boundary nodes
        auto indices = std::get<0>(this->mesh->boundaryNodes(boundaryElement));
        auto points = std::get<1>(this->mesh->boundaryNodes(boundaryElement));

        // sort nodes by parameter
        std::vector<std::tuple<dtype::real, dtype::real>> nodes(points.rows());
        for (dtype::index i = 0; i < points.rows(); ++i) {
            nodes[i] = std::make_tuple(points(i, 0), points(i, 1));
        }
        std::sort(nodes.begin(), nodes.end(),
            [](const std::tuple<dtype::real, dtype::real>& a,
                const std::tuple<dtype::real, dtype::real>& b)
                -> bool {
                    return math::circleParameter(b, math::circleParameter(a, 0.0)) > 0.0;
        });

        // calc parameter offset
        dtype::real parameterOffset = math::circleParameter(nodes[0], 0.0);

        // calc node parameter centered to node 0
        Eigen::ArrayXf nodeParameter = Eigen::ArrayXf::Zero(points.rows());
        for (dtype::size i = 0; i < nodes.size(); ++i) {
            nodeParameter(i) = math::circleParameter(nodes[i], parameterOffset);
        }

        for (dtype::index piece = 0; piece < this->boundaryDescriptor->count; ++piece) {
            // skip boundary part, if radii dont match
            if (std::abs(
                std::get<0>(math::polar(std::get<0>(this->boundaryDescriptor->coordinates[piece]))) -
                std::get<0>(math::polar(nodes[0]))) > 0.0001) {
                continue;
            }

            // calc integration interval centered to node 0
            auto integrationStart = math::circleParameter(
                std::get<0>(this->boundaryDescriptor->coordinates[piece]),
                parameterOffset);
            auto integrationEnd = math::circleParameter(
                std::get<1>(this->boundaryDescriptor->coordinates[piece]),
                parameterOffset);

            // intgrate if integrationStart is left of integrationEnd
            if (integrationStart < integrationEnd) {
                // calc element
                for (dtype::index node = 0; node < basisFunctionType::pointsPerEdge; ++node) {
                    (*excitationMatrix)(indices(node), piece) +=
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
    // fill connectivity and elementalJacobianMatrix
    auto elementalJacobianMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->elementalJacobianMatrix->rows, this->elementalJacobianMatrix->cols, stream);
    for (dtype::index element = 0; element < this->mesh->elements->rows; ++element) {
        // get element nodes
        auto points = std::get<1>(this->mesh->elementNodes(element));

        // fill matrix
        for (dtype::index i = 0; i < basisFunctionType::pointsPerElement; ++i)
        for (dtype::index j = 0; j < basisFunctionType::pointsPerElement; ++j) {
            // create basis functions
            auto basisI = std::make_shared<basisFunctionType>(points, i);
            auto basisJ = std::make_shared<basisFunctionType>(points, j);

            // set elementalJacobianMatrix element
            (*elementalJacobianMatrix)(element, i +
                j * basisFunctionType::pointsPerElement) =
                basisI->integrateGradientWithBasis(basisJ);
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
        numeric::matrix::block_size, stream, this->sMatrix->values, this->rMatrix->values,
        this->sMatrix->columnIds, this->sMatrix->density, k, this->systemMatrix->values);
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
    FEM::equationKernel::calcJacobian<dataType, basisFunctionType::pointsPerElement>(blocks, threads, stream,
        phi->deviceData, &phi->deviceData[driveCount * phi->dataRows],
        this->mesh->elements->deviceData, this->elementalJacobianMatrix->deviceData,
        gamma->deviceData, this->referenceValue, jacobian->dataRows, jacobian->dataCols,
        phi->dataRows, this->mesh->elements->rows, driveCount, measurmentCount, additiv,
        jacobian->deviceData);
}

// reduce matrix
template <
    class inputType,
    class shapeType,
    class outputType
>
void mpFlow::FEM::equation::reduceMatrix(
    const std::shared_ptr<numeric::Matrix<inputType>> intermediateMatrix,
    const std::shared_ptr<numeric::SparseMatrix<shapeType>> shape, dtype::index offset,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<outputType>> matrix) {
    // check input
    if (intermediateMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::reduceMatrix: intermediateMatrix == nullptr");
    }
    if (shape == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::reduceMatrix: shape == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::FEM::equation::reduceMatrix: matrix == nullptr");
    }

    // block size
    dim3 blocks(matrix->dataRows / numeric::matrix::block_size, 1);
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);

    // reduce matrix
    FEM::equationKernel::reduceMatrix(blocks, threads, stream,
        intermediateMatrix->deviceData, shape->columnIds, matrix->dataRows,
        offset, matrix->deviceData);
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
        referenceValue, connectivityMatrix->dataRows, connectivityMatrix->dataCols, matrix->values);
}

// specialisation
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::index, mpFlow::dtype::real, mpFlow::dtype::index>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::index, mpFlow::dtype::complex, mpFlow::dtype::index>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::complex>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::real, mpFlow::dtype::real, mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::real, mpFlow::dtype::complex, mpFlow::dtype::complex>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::complex>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::complex>>);

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
// template class mpFlow::FEM::Equation<mpFlow::dtype::complex, mpFlow::FEM::basis::Edge>;
