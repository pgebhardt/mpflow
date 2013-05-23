// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fasteit/cuda_error.h"

#include "fasteit/dtype.h"
#include "fasteit/constants.h"
#include "fasteit/sparse_matrix_kernel.h"

// convert to sparse matrix kernel
template <
    class type
>
static __global__ void convertKernel(const type* matrix,
    fastEIT::dtype::size rows, fastEIT::dtype::size columns,
    type* values, fastEIT::dtype::index* columnIds,
    fastEIT::dtype::index* elementCount) {
    // get id
    fastEIT::dtype::index i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    fastEIT::dtype::size count = 0;

    // init values and columnIds
    for (fastEIT::dtype::index j = 0; j < fastEIT::sparseMatrix::block_size; j++) {
        values[i * fastEIT::sparseMatrix::block_size + j] = 0.0f;
        columnIds[i * fastEIT::sparseMatrix::block_size + j] = fastEIT::dtype::invalid_index;
    }

    // search non-zero elements
    type element = 0.0f;
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
template <
    class type
>
void fastEIT::sparseMatrixKernel::convert(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* matrix, dtype::size rows, dtype::size columns,
    type* values, dtype::index* columnIds, dtype::index* elementCount) {
    // call cuda kernel
    convertKernel<type><<<blocks, threads, 0, stream>>>(matrix, rows, columns,
        values, columnIds, elementCount);

    CudaCheckError();
}

// convert to matrix kernel
template <
    class type
>
static __global__ void convertToMatrixKernel(const type* values,
    const fastEIT::dtype::index* column_ids, fastEIT::dtype::size density,
    fastEIT::dtype::size rows, type* matrix) {
    // get row id
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // expand sparse matrix
    fastEIT::dtype::index column_id = fastEIT::dtype::invalid_index;
    for (fastEIT::dtype::index column = 0; column < density; ++column) {
        // get column id
        column_id = column_ids[row * fastEIT::sparseMatrix::block_size + column];

        // set matrix value
        if (column_id != fastEIT::dtype::invalid_index) {
            matrix[row + column_id * rows] = values[
                row * fastEIT::sparseMatrix::block_size + column];
        }
    }
}

// convert to matrix kernel wrapper
template <
    class type
>
void fastEIT::sparseMatrixKernel::convertToMatrix(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* values, const dtype::index* column_ids,
    dtype::size density, dtype::size rows, type* matrix) {
    // call cuda kernel
    convertToMatrixKernel<type><<<blocks, threads, 0, stream>>>(values, column_ids,
        density, rows, matrix);

    CudaCheckError();
}

// sparse matrix multiply kernel
template <
    class type
>
static __global__ void multiplyKernel(const type* values,
    const fastEIT::dtype::index* columnIds, const type* matrix,
    fastEIT::dtype::size result_rows, fastEIT::dtype::size matrix_rows,
    fastEIT::dtype::size columns, fastEIT::dtype::size density, type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    type res = 0.0f;
    fastEIT::dtype::index id = fastEIT::dtype::invalid_index;

    // read column ids to local memory
    __shared__ fastEIT::dtype::index columnId[
        fastEIT::sparseMatrix::block_size * fastEIT::sparseMatrix::block_size];
    __shared__ type value[
        fastEIT::sparseMatrix::block_size * fastEIT::sparseMatrix::block_size];

    columnId[threadIdx.x * fastEIT::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        columnIds[row * fastEIT::sparseMatrix::block_size + threadIdx.y] : fastEIT::dtype::invalid_index;
    value[threadIdx.x * fastEIT::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        values[row * fastEIT::sparseMatrix::block_size + threadIdx.y] : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= result_rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (fastEIT::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * fastEIT::sparseMatrix::block_size + j];

         res += id != fastEIT::dtype::invalid_index ? matrix[id + column * matrix_rows] *
            value[threadIdx.x * fastEIT::sparseMatrix::block_size + j] : 0.0f;
    }

    // set result
    result[row + column * result_rows] = res;
}

// sparse matrix multiply kernel wrapper
template <
    class type
>
void fastEIT::sparseMatrixKernel::multiply(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* values, const dtype::index* columnIds,
    const type* matrix, dtype::size result_rows, dtype::size matrix_rows,
    dtype::size columns, dtype::size density, type* result) {
    // call cuda kernel
    multiplyKernel<type><<<blocks, threads, 0, stream>>>(values, columnIds, matrix,
        result_rows, matrix_rows, columns, density, result);

    CudaCheckError();
}

// specialisations
// convert to sparse matrix kernel
template void fastEIT::sparseMatrixKernel::convert<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, fastEIT::dtype::size, fastEIT::dtype::size,
    fastEIT::dtype::real*, fastEIT::dtype::index*, fastEIT::dtype::index*);
template void fastEIT::sparseMatrixKernel::convert<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, fastEIT::dtype::size, fastEIT::dtype::size,
    fastEIT::dtype::index*, fastEIT::dtype::index*, fastEIT::dtype::index*);

// convertToMatrix kernel
template void fastEIT::sparseMatrixKernel::convertToMatrix<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::size, fastEIT::dtype::real* matrix);
template void fastEIT::sparseMatrixKernel::convertToMatrix<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::size, fastEIT::dtype::index* matrix);

// multiply kernel
template void fastEIT::sparseMatrixKernel::multiply<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, const fastEIT::dtype::index*,
    const fastEIT::dtype::real*, fastEIT::dtype::size, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::sparseMatrixKernel::multiply<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, const fastEIT::dtype::index*,
    const fastEIT::dtype::index*, fastEIT::dtype::size, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::size, fastEIT::dtype::index*);
