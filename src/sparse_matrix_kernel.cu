// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/constants.h"
#include "mpflow/sparse_matrix_kernel.h"

// convert to sparse matrix kernel
template <
    class type
>
static __global__ void convertKernel(const type* matrix,
    mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    type* values, mpFlow::dtype::index* columnIds,
    mpFlow::dtype::index* elementCount) {
    // get id
    mpFlow::dtype::index i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    mpFlow::dtype::size count = 0;

    // init values and columnIds
    for (mpFlow::dtype::index j = 0; j < mpFlow::sparseMatrix::block_size; j++) {
        values[i * mpFlow::sparseMatrix::block_size + j] = 0.0f;
        columnIds[i * mpFlow::sparseMatrix::block_size + j] = mpFlow::dtype::invalid_index;
    }

    // search non-zero elements
    type element = 0.0f;
    for (mpFlow::dtype::index j = 0; j < columns; j++) {
        // get element
        element = matrix[i + j * rows];

        // check for non-zero
        if (element != 0.0f) {
            values[i * mpFlow::sparseMatrix::block_size + count] = element;
            columnIds[i * mpFlow::sparseMatrix::block_size + count] = j;

            // increment count
            count++;

            // check count
            if (count >= mpFlow::sparseMatrix::block_size) {
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
void mpFlow::sparseMatrixKernel::convert(dim3 blocks, dim3 threads, cudaStream_t stream,
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
    const mpFlow::dtype::index* column_ids, mpFlow::dtype::size density,
    mpFlow::dtype::size rows, type* matrix) {
    // get row id
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // expand sparse matrix
    mpFlow::dtype::index column_id = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0; column < density; ++column) {
        // get column id
        column_id = column_ids[row * mpFlow::sparseMatrix::block_size + column];

        // set matrix value
        if (column_id != mpFlow::dtype::invalid_index) {
            matrix[row + column_id * rows] = values[
                row * mpFlow::sparseMatrix::block_size + column];
        }
    }
}

// convert to matrix kernel wrapper
template <
    class type
>
void mpFlow::sparseMatrixKernel::convertToMatrix(dim3 blocks, dim3 threads,
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
    const mpFlow::dtype::index* columnIds, const type* matrix,
    mpFlow::dtype::size result_rows, mpFlow::dtype::size matrix_rows,
    mpFlow::dtype::size columns, mpFlow::dtype::size density, type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    type res = 0.0f;
    mpFlow::dtype::index id = mpFlow::dtype::invalid_index;

    // read column ids to local memory
    __shared__ mpFlow::dtype::index columnId[
        mpFlow::sparseMatrix::block_size * mpFlow::sparseMatrix::block_size];
    __shared__ type value[
        mpFlow::sparseMatrix::block_size * mpFlow::sparseMatrix::block_size];

    columnId[threadIdx.x * mpFlow::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        columnIds[row * mpFlow::sparseMatrix::block_size + threadIdx.y] : mpFlow::dtype::invalid_index;
    value[threadIdx.x * mpFlow::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        values[row * mpFlow::sparseMatrix::block_size + threadIdx.y] : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= result_rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (mpFlow::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * mpFlow::sparseMatrix::block_size + j];

         res += id != mpFlow::dtype::invalid_index ? matrix[id + column * matrix_rows] *
            value[threadIdx.x * mpFlow::sparseMatrix::block_size + j] : 0.0f;
    }

    // set result
    result[row + column * result_rows] = res;
}

// sparse matrix multiply kernel wrapper
template <
    class type
>
void mpFlow::sparseMatrixKernel::multiply(dim3 blocks, dim3 threads, cudaStream_t stream,
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
template void mpFlow::sparseMatrixKernel::convert<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::real*, mpFlow::dtype::index*, mpFlow::dtype::index*);
template void mpFlow::sparseMatrixKernel::convert<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::index*, mpFlow::dtype::index*, mpFlow::dtype::index*);

// convertToMatrix kernel
template void mpFlow::sparseMatrixKernel::convertToMatrix<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, mpFlow::dtype::real* matrix);
template void mpFlow::sparseMatrixKernel::convertToMatrix<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, mpFlow::dtype::index* matrix);

// multiply kernel
template void mpFlow::sparseMatrixKernel::multiply<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    const mpFlow::dtype::real*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::sparseMatrixKernel::multiply<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, mpFlow::dtype::index*);
