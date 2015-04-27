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
#include "mpflow/numeric/preconditioner_kernel.h"

// diagonal preconditioner
template <class type>
void mpFlow::numeric::preconditioner::diagonal(std::shared_ptr<Matrix<type> const> const matrix,
    cudaStream_t const stream, std::shared_ptr<SparseMatrix<type>> const result) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::preconditioner::diagonal: matrix == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::preconditioner::diagonal: result == nullptr");
    }

    dim3 blocks(matrix->dataRows / numeric::matrix::blockSize, 1);
    dim3 threads(numeric::matrix::blockSize, 1);

    preconditionerKernel::diagonal<type>(blocks, threads, stream,
        matrix->deviceData, matrix->dataRows,
        result->deviceValues, result->deviceColumnIds);
    result->density = 1;
}

template void mpFlow::numeric::preconditioner::diagonal<float>(
    std::shared_ptr<mpFlow::numeric::Matrix<float> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const);
template void mpFlow::numeric::preconditioner::diagonal<double>(
    std::shared_ptr<mpFlow::numeric::Matrix<double> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const);
template void mpFlow::numeric::preconditioner::diagonal<thrust::complex<float>>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const);
template void mpFlow::numeric::preconditioner::diagonal<thrust::complex<double>>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const);

template <class type>
void mpFlow::numeric::preconditioner::diagonal(std::shared_ptr<SparseMatrix<type> const> const matrix,
    cudaStream_t const stream, std::shared_ptr<SparseMatrix<type>> const result) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::preconditioner::diagonal: matrix == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::preconditioner::diagonal: result == nullptr");
    }

    dim3 blocks(matrix->dataRows / numeric::matrix::blockSize, 1);
    dim3 threads(numeric::matrix::blockSize, 1);

    preconditionerKernel::diagonalSparse<type>(blocks, threads, stream,
        matrix->deviceValues, matrix->deviceColumnIds,
        result->deviceValues, result->deviceColumnIds);
    result->density = 1;
}

template void mpFlow::numeric::preconditioner::diagonal<float>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const);
template void mpFlow::numeric::preconditioner::diagonal<double>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const);
template void mpFlow::numeric::preconditioner::diagonal<thrust::complex<float>>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const);
template void mpFlow::numeric::preconditioner::diagonal<thrust::complex<double>>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>> const> const, cudaStream_t const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const);
