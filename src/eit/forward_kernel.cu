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
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/eit/forward_kernel.h"

// calc voltage kernel
static __global__ void applyMixedBoundaryConditionKernel(
    mpFlow::dtype::real* excitation, const mpFlow::dtype::index* columnIds,
    mpFlow::dtype::real* values) {
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // clip excitation value
    excitation[row] = excitation[row] > 0.0f ? 1.0f : (excitation[row] < 0.0f ? -1.0f : 0.0);

    // clear matrix row and set diagonal element to 1, if excitation element != 0
    mpFlow::dtype::index columnId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0;
        column < mpFlow::numeric::sparseMatrix::block_size;
        ++column) {
        // get column id
        columnId = columnIds[row * mpFlow::numeric::sparseMatrix::block_size + column];

        values[row * mpFlow::numeric::sparseMatrix::block_size + column] =
            excitation[row] != 0.0f ? (columnId == row ? 1.0f : 0.0f) :
            values[row * mpFlow::numeric::sparseMatrix::block_size + column];
    }
}

// calc voltage kernel wrapper
void mpFlow::EIT::forwardKernel::applyMixedBoundaryCondition(
    dim3 blocks, dim3 threads, cudaStream_t stream,
    dtype::real* excitation, const dtype::index* columnIds,
    dtype::real* values) {
    // call cuda kernel
    applyMixedBoundaryConditionKernel<<<blocks, threads, 0, stream>>>(
        excitation, columnIds, values);

    CudaCheckError();
}
