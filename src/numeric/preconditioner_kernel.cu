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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include "mpflow/cuda_error.h"

#include "mpflow/constants.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/numeric/preconditioner_kernel.h"

template <class type>
__global__ void diagonalKernel(type const* const data, unsigned const rows,
    type* const outputValues, unsigned* const outputColumnIds) {
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;

    // update gc
    for (int i = 0; i < mpFlow::numeric::sparseMatrix::blockSize; ++i) {
        outputColumnIds[row * mpFlow::numeric::sparseMatrix::blockSize + i] = i == 0 ? row : mpFlow::constants::invalidIndex;
    }

    outputValues[row * mpFlow::numeric::sparseMatrix::blockSize] = type(1) / data[row + row * rows];
}

template <class type>
void mpFlow::numeric::preconditionerKernel::diagonal(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    type const* const data, unsigned const rows,
    type* const outputValues, unsigned* const outputColumnIds) {
    diagonalKernel<type><<<blocks, threads, 0, stream>>>(
        data, rows, outputValues, outputColumnIds);

    CudaCheckError();
}

template void mpFlow::numeric::preconditionerKernel::diagonal<float>(dim3 const, dim3 const, cudaStream_t const,
    float const* const, unsigned const, float* const, unsigned* const);
template void mpFlow::numeric::preconditionerKernel::diagonal<double>(dim3 const, dim3 const, cudaStream_t const,
    double const* const, unsigned const, double* const, unsigned* const);
template void mpFlow::numeric::preconditionerKernel::diagonal<thrust::complex<float> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<float> const* const, unsigned const, thrust::complex<float>* const, unsigned* const);
template void mpFlow::numeric::preconditionerKernel::diagonal<thrust::complex<double> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<double> const* const, unsigned const, thrust::complex<double>* const, unsigned* const);

template <class type>
__global__ void diagonalSparseKernel(type const* const values, unsigned const* const columnIds,
    type* const outputValues, unsigned* const outputColumnIds) {
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;

    // update gc
    for (int i = 0; i < mpFlow::numeric::sparseMatrix::blockSize; ++i) {
        outputColumnIds[row * mpFlow::numeric::sparseMatrix::blockSize + i] = i == 0 ? row : mpFlow::constants::invalidIndex;
    }

    // search for diagonal element
    outputValues[row * mpFlow::numeric::sparseMatrix::blockSize] = type(0);
    for (int i = 0; i < mpFlow::numeric::sparseMatrix::blockSize; ++i) {
        unsigned const columnId = columnIds[row * mpFlow::numeric::sparseMatrix::blockSize + i];

        if (columnId == row) {
            outputValues[row * mpFlow::numeric::sparseMatrix::blockSize] =
                type(1) / values[row * mpFlow::numeric::sparseMatrix::blockSize + i];
            break;
        }
    }
}

template <class type>
void mpFlow::numeric::preconditionerKernel::diagonalSparse(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    type const* const values, unsigned const* const columnIds,
    type* const outputValues, unsigned* const outputColumnIds) {
    diagonalSparseKernel<type><<<blocks, threads, 0, stream>>>(
        values, columnIds, outputValues, outputColumnIds);

    CudaCheckError();
}

template void mpFlow::numeric::preconditionerKernel::diagonalSparse<float>(dim3 const, dim3 const, cudaStream_t const,
    float const* const, unsigned const* const, float* const, unsigned* const);
template void mpFlow::numeric::preconditionerKernel::diagonalSparse<double>(dim3 const, dim3 const, cudaStream_t const,
    double const* const, unsigned const* const, double* const, unsigned* const);
template void mpFlow::numeric::preconditionerKernel::diagonalSparse<thrust::complex<float> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<float> const* const, unsigned const* const, thrust::complex<float>* const, unsigned* const);
template void mpFlow::numeric::preconditionerKernel::diagonalSparse<thrust::complex<double> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<double> const* const, unsigned const* const, thrust::complex<double>* const, unsigned* const);
