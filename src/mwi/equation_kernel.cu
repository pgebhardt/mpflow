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

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/mwi/equation_kernel.h"

// assemble complex system matrix
static __global__ void assembleComplexSystemKernel(
    const mpFlow::dtype::real* realValues, const mpFlow::dtype::index* realColumnIds,
    mpFlow::dtype::index realRows, const mpFlow::dtype::real* imaginaryValues,
    mpFlow::dtype::real* completeValues, mpFlow::dtype::index* completeColumnIds) {

    // get id
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // upper part of matrix (Ar  -Ai)
    if (row < realRows) {
        for (mpFlow::dtype::index col = 0; col < mpFlow::numeric::sparseMatrix::block_size; ++col) {
            mpFlow::dtype::index columnId = realColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + col];

            if (columnId != mpFlow::dtype::invalid_index) {
                completeValues[row * mpFlow::numeric::sparseMatrix::block_size + col] =
                    realValues[row * mpFlow::numeric::sparseMatrix::block_size + col];
                completeColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + col] = columnId;
            }
            else {
                completeValues[row * mpFlow::numeric::sparseMatrix::block_size + col] = -imaginaryValues[row];
                completeColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + col] = realRows + row;

                break;
            }
        }
    }

    // lower part of matrix (Ai  Ar)
    else {
        for (mpFlow::dtype::index col = 0; col < mpFlow::numeric::sparseMatrix::block_size; ++col) {

            if (col == 0) {
                completeValues[row * mpFlow::numeric::sparseMatrix::block_size + col] = imaginaryValues[row - realRows];
                completeColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + col] = row - realRows;
            }
            else {
                completeValues[row * mpFlow::numeric::sparseMatrix::block_size + col] =
                    realValues[(row - realRows) * mpFlow::numeric::sparseMatrix::block_size + col - 1];
                completeColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + col] =
                    realColumnIds[(row - realRows) * mpFlow::numeric::sparseMatrix::block_size + col - 1] + realRows;
            }
        }
    }
}

// assemble complex system matrix wrapper
void mpFlow::MWI::equationKernel::assembleComplexSystem(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::real* realValues, const dtype::index* realColumnIds, dtype::index realRows,
    const dtype::real* imaginaryValues, dtype::real* completeValues, dtype::index* completeColumnIds) {
    // call cuda kernel
    assembleComplexSystemKernel<<<blocks, threads, 0, stream>>>(realValues, realColumnIds, realRows,
        imaginaryValues, completeValues, completeColumnIds);

    CudaCheckError();
}
