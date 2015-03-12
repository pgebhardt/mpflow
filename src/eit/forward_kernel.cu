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

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/eit/forward_kernel.h"

// calc voltage kernel
template <
    class dataType
>
static __global__ void applyMixedBoundaryConditionKernel(
    dataType* excitation, mpFlow::dtype::index rows,
    const mpFlow::dtype::index* columnIds, dataType* values) {
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index col = blockIdx.y * blockDim.y + threadIdx.y;

    // clip excitation value
    excitation[row + col * rows] = abs(excitation[row + col * rows]) >= 1e-9 ? 1.0f : 0.0f;

    // clear matrix row and set diagonal element to 1, if excitation element != 0
    mpFlow::dtype::index columnId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0;
        column < mpFlow::numeric::sparseMatrix::block_size;
        ++column) {
        // get column id
        columnId = columnIds[row * mpFlow::numeric::sparseMatrix::block_size + column];

        if (excitation[row + col * rows] != (mpFlow::dtype::real)0.0) {
            values[row * mpFlow::numeric::sparseMatrix::block_size + column] =
                columnId == row ? 1.0f : 0.0f;
        }
    }
}

// calc voltage kernel wrapper
template <
    class dataType
>
void mpFlow::EIT::forwardKernel::applyMixedBoundaryCondition(
    dim3 blocks, dim3 threads, cudaStream_t stream,
    dataType* excitation, dtype::index rows,
    const dtype::index* columnIds, dataType* values) {
    // call cuda kernel
    applyMixedBoundaryConditionKernel<dataType><<<blocks, threads, 0, stream>>>(
        excitation, rows, columnIds, values);

    CudaCheckError();
}

template void mpFlow::EIT::forwardKernel::applyMixedBoundaryCondition<mpFlow::dtype::real>(
    dim3, dim3, cudaStream_t, mpFlow::dtype::real*, mpFlow::dtype::index,
    const mpFlow::dtype::index*, mpFlow::dtype::real*);
template void mpFlow::EIT::forwardKernel::applyMixedBoundaryCondition<mpFlow::dtype::complex>(
    dim3, dim3, cudaStream_t, mpFlow::dtype::complex*, mpFlow::dtype::index,
    const mpFlow::dtype::index*, mpFlow::dtype::complex*);
