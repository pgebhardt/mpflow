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

// reduce matrix
template <
    class dataType,
    class shapeDataType
>
void mpFlow::FEM::equation::reduceMatrix(
    const std::shared_ptr<numeric::Matrix<dataType>> intermediateMatrix,
    const std::shared_ptr<numeric::SparseMatrix<shapeDataType>> shape, dtype::index offset,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<dataType>> matrix) {
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
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::real, mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::index, mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::complex, mpFlow::dtype::complex>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::complex>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::complex>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::complex>>);
template void mpFlow::FEM::equation::reduceMatrix<mpFlow::dtype::index, mpFlow::dtype::complex>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::complex>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);

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
