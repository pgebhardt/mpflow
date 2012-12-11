// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/constants.h"
#include "../include/sparse_matrix_kernel.h"

// convert to sparse matrix kernel
static __global__ void convertKernel(const fastEIT::dtype::real* matrix,
    fastEIT::dtype::size rows, fastEIT::dtype::size columns,
    fastEIT::dtype::real* values, fastEIT::dtype::index* columnIds,
    fastEIT::dtype::index* elementCount) {
    // get id
    fastEIT::dtype::index i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    fastEIT::dtype::size count = 0;

    // init values and columnIds
    for (fastEIT::dtype::index j = 0; j < fastEIT::sparseMatrix::block_size; j++) {
        values[i * fastEIT::sparseMatrix::block_size + j] = 0.0f;
        columnIds[i * fastEIT::sparseMatrix::block_size + j] = -1;
    }

    // search non-zero elements
    fastEIT::dtype::real element = 0.0f;
    for (fastEIT::dtype::index j = 0; j < columns; j++) {
        // get element
        element = matrix[i + j * rows];

        // check for non-zero
        if (element != 0.0f) {
            values[i * fastEIT::sparseMatrix::block_size + count] = element;
            columnIds[i * fastEIT::sparseMatrix::block_size + count] = j;

            // increment count
            count++;

            // check count
            if (count >= fastEIT::sparseMatrix::block_size) {
                break;
            }
        }
    }

    // save element count
    elementCount[i] = count;
}

// convert to sparse matrix kernel wrapper
void fastEIT::sparseMatrixKernel::convert(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::real* matrix, dtype::size rows, dtype::size columns,
    dtype::real* values, dtype::index* columnIds, dtype::index* elementCount) {
    // call cuda kernel
    convertKernel<<<blocks, threads, 0, stream>>>(matrix, rows, columns,
        values, columnIds, elementCount);
}

// sparse matrix multiply kernel
static __global__ void multiplyKernel(const fastEIT::dtype::real* values,
    const fastEIT::dtype::index* columnIds, const fastEIT::dtype::real* matrix,
    fastEIT::dtype::size rows, fastEIT::dtype::size columns,
    fastEIT::dtype::size density, fastEIT::dtype::real* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    fastEIT::dtype::real res = 0.0f;
    fastEIT::dtype::index id = -1;

    // read column ids to local memory
    __shared__ fastEIT::dtype::index columnId[
        fastEIT::sparseMatrix::block_size * fastEIT::sparseMatrix::block_size];
    __shared__ fastEIT::dtype::real value[
        fastEIT::sparseMatrix::block_size * fastEIT::sparseMatrix::block_size];

    columnId[threadIdx.x * fastEIT::sparseMatrix::block_size + threadIdx.y] = row < rows ?
        columnIds[row * fastEIT::sparseMatrix::block_size + threadIdx.y] : -1;
    value[threadIdx.x * fastEIT::sparseMatrix::block_size + threadIdx.y] = row < rows ?
        values[row * fastEIT::sparseMatrix::block_size + threadIdx.y] : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (fastEIT::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * fastEIT::sparseMatrix::block_size + j];

         res += id != -1 ? matrix[id + column * rows] *
            value[threadIdx.x * fastEIT::sparseMatrix::block_size + j] : 0.0f;
    }

    // set result
    result[row + column * rows] = res;
}

// sparse matrix multiply kernel wrapper
void fastEIT::sparseMatrixKernel::multiply(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::real* values, const dtype::index* columnIds,
    const dtype::real* matrix, dtype::size rows, dtype::size columns,
    dtype::size density, dtype::real* result) {
    // call cuda kernel
    multiplyKernel<<<blocks, threads, 0, stream>>>(values, columnIds, matrix,
        rows, columns, density, result);
}
