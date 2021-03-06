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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include "mpflow/cuda_error.h"

#include "mpflow/constants.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/models/eit_kernel.h"

// calc voltage kernel
template <
    class dataType
>
static __global__ void applyMixedBoundaryConditionKernel(
    dataType* const excitation, unsigned const rows,
    unsigned const* const columnIds, dataType* const values) {
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned const col = blockIdx.y * blockDim.y + threadIdx.y;

    // clip excitation value
    excitation[row + col * rows] = abs(excitation[row + col * rows]) >= 1e-9 ? dataType(1) : dataType(0);

    // clear matrix row and set diagonal element to 1, if excitation element != 0
    unsigned columnId = mpFlow::constants::invalidIndex;
    for (unsigned column = 0; column < mpFlow::numeric::sparseMatrix::blockSize; ++column) {
        // get column id
        columnId = columnIds[row * mpFlow::numeric::sparseMatrix::blockSize + column];

        if (excitation[row + col * rows] != dataType(0)) {
            values[row * mpFlow::numeric::sparseMatrix::blockSize + column] =
                columnId == row ? dataType(1) : dataType(0);
        }
    }
}

// calc voltage kernel wrapper
template <
    class dataType
>
void mpFlow::models::eitKernel::applyMixedBoundaryCondition(
    dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    dataType* const excitation, unsigned const rows,
    unsigned const* const columnIds, dataType* const values) {
    // call cuda kernel
    applyMixedBoundaryConditionKernel<dataType><<<blocks, threads, 0, stream>>>(
        excitation, rows, columnIds, values);

    CudaCheckError();
}

template void mpFlow::models::eitKernel::applyMixedBoundaryCondition<float>(
    dim3 const, dim3 const, cudaStream_t const, float* const, unsigned const,
    unsigned const* const, float* const);
template void mpFlow::models::eitKernel::applyMixedBoundaryCondition<double>(
    dim3 const, dim3 const, cudaStream_t const, double* const, unsigned const,
    unsigned const* const, double* const);
template void mpFlow::models::eitKernel::applyMixedBoundaryCondition<thrust::complex<float> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<float>* const, unsigned const,
    unsigned const* const, thrust::complex<float>* const);
template void mpFlow::models::eitKernel::applyMixedBoundaryCondition<thrust::complex<double> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<double>* const, unsigned const,
    unsigned const* const, thrust::complex<double>* const);
