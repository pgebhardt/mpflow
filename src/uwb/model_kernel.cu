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
#include "mpflow/uwb/model_kernel.h"

// reduce connectivity and elementalResidual matrix
template <
    class type
>
static __global__ void reduceMatrixKernel(const type* intermediate_matrix,
    const mpFlow::dtype::index* column_ids, mpFlow::dtype::size rows,
    mpFlow::dtype::index offset, type* matrix) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    mpFlow::dtype::index columnId = column_ids[row * mpFlow::numeric::sparseMatrix::block_size + column];

    // check column id
    if (columnId == mpFlow::dtype::invalid_index) {
        return;
    }

    // reduce matrices
    matrix[row + (column + offset * mpFlow::numeric::sparseMatrix::block_size) * rows] =
        intermediate_matrix[row + columnId * rows];
}

// reduce matrix wrapper
template <
    class type
>
void mpFlow::UWB::modelKernel::reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* intermediate_matrix, const dtype::index* column_ids, dtype::size rows,
    dtype::index offset, type* matrix) {
    // call cuda kernel
    reduceMatrixKernel<type><<<blocks, threads, 0, stream>>>(intermediate_matrix,
        column_ids, rows, offset, matrix);

    CudaCheckError();
}

// reduce matrix specialisation
template void mpFlow::UWB::modelKernel::reduceMatrix<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::real*);
template void mpFlow::UWB::modelKernel::reduceMatrix<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::index*);

// update matrix kernel
static __global__ void updateMatrixKernel(const mpFlow::dtype::index* connectivityMatrix,
    const mpFlow::dtype::real* elementalMatrix, const mpFlow::dtype::real* material,
    mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    mpFlow::dtype::real* matrix_values) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    mpFlow::dtype::real value = 0.0f;
    mpFlow::dtype::index elementId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index k = 0; k < columns / mpFlow::numeric::sparseMatrix::block_size; ++k) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows];

        value += elementId != mpFlow::dtype::invalid_index ? elementalMatrix[row +
            (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows] *
            material[elementId] : 0.0f;
    }

    // set residual matrix element
    matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] = value;
}

// update matrix kernel wrapper
void mpFlow::UWB::modelKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::index* connectivityMatrix, const dtype::real* elementalMatrix,
    const dtype::real* material, dtype::size rows, dtype::size columns,
    dtype::real* matrix_values) {
    // call cuda kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(connectivityMatrix, elementalMatrix,
        material, rows, columns, matrix_values);

    CudaCheckError();
}

// update system matrix kernel
static __global__ void updateSystemMatrixKernel(
    const mpFlow::dtype::real* sMatrixValues, const mpFlow::dtype::real* rMatrixValues,
    const mpFlow::dtype::index* sMatrixColumnIds, mpFlow::dtype::size density,
    mpFlow::dtype::real sScalar, mpFlow::dtype::real rScalar,
    mpFlow::dtype::index rowOffset, mpFlow::dtype::index columnOffset,
    mpFlow::dtype::real* systemMatrixValues) {
    // get row
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // update system matrix
    mpFlow::dtype::index columnId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0; column < density; ++column) {
        // get column id
        columnId = sMatrixColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + column];

        // update system matrix element
        systemMatrixValues[(row + rowOffset) * mpFlow::numeric::sparseMatrix::block_size + column + columnOffset] =
            columnId != mpFlow::dtype::invalid_index ?
            sMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column] * sScalar +
            rMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column] * rScalar :
            systemMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column];
    }
}

// update system matrix kernel wrapper
void mpFlow::UWB::modelKernel::updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const mpFlow::dtype::real* sMatrixValues, const mpFlow::dtype::real* rMatrixValues,
    const mpFlow::dtype::index* sMatrixColumnIds, mpFlow::dtype::size density,
    mpFlow::dtype::real sScalar, mpFlow::dtype::real rScalar,
    mpFlow::dtype::index rowOffset, mpFlow::dtype::index columnOffset,
    mpFlow::dtype::real* systemMatrixValues) {
    // call cuda kernel
    updateSystemMatrixKernel<<<blocks, threads, 0, stream>>>(
        sMatrixValues, rMatrixValues, sMatrixColumnIds, density,
        sScalar, rScalar, rowOffset, columnOffset, systemMatrixValues);

    CudaCheckError();
}
