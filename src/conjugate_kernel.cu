// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/constants.h"
#include "../include/conjugate_kernel.h"

// add scalar kernel
static __global__ void addScalarKernel(const fastEIT::dtype::real* scalar,
    fastEIT::dtype::size vectorRows, fastEIT::dtype::size rows,
    fastEIT::dtype::size columns, fastEIT::dtype::real* vector) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vectorRows] += row < rows && column < columns ?
        scalar[column * vectorRows] : 0.0f;
}

// add scalar kernel wrapper
void fastEIT::numeric::conjugateKernel::addScalar(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dtype::real* scalar, dtype::size vector_rows,
    dtype::size rows, dtype::size columns, dtype::real* vector) {
    // call cuda kernel
    addScalarKernel<<<blocks, threads, 0, stream>>>(scalar, vector_rows,
        rows, columns, vector);
}

// update vector kernel
static __global__ void updateVectorKernel(const fastEIT::dtype::real* x1,
    const fastEIT::dtype::real sign, const fastEIT::dtype::real* x2,
    const fastEIT::dtype::real* r1, const fastEIT::dtype::real* r2,
    fastEIT::dtype::size rows, fastEIT::dtype::real* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] +
        sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector kernel wrapper
void fastEIT::numeric::conjugateKernel::updateVector(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dtype::real* x1, const dtype::real sign,
    const dtype::real* x2, const dtype::real* r1, const dtype::real* r2,
    dtype::size rows, dtype::real* result) {
    // call cuda kernel
    updateVectorKernel<<<blocks, threads, 0, stream>>>(x1, sign, x2, r1, r2, rows,
        result);
}

// gemv kernel
static __global__ void gemvKernel(const fastEIT::dtype::real* matrix,
    const fastEIT::dtype::real* vector, fastEIT::dtype::size rows,
    fastEIT::dtype::real* result) {
    // get ids
    fastEIT::dtype::index row = threadIdx.x + blockIdx.x * blockDim.x;
    fastEIT::dtype::index column = (threadIdx.y + blockIdx.y * blockDim.y) *
        2 * fastEIT::matrix::block_size;

    // load vector to shared memory
    __shared__ fastEIT::dtype::real work[2 * fastEIT::matrix::block_size *
        fastEIT::matrix::block_size];
    work[threadIdx.x +
        threadIdx.y * 2 * fastEIT::matrix::block_size] =
        column + threadIdx.x < rows ? vector[column + threadIdx.x] : 0.0f;
    __syncthreads();

    // compute partial vector product
    fastEIT::dtype::real product = 0.0f;
    for (fastEIT::dtype::index i = 0; i < 2 * fastEIT::matrix::block_size; i++) {
        product += row < rows && column + i < rows ?
            matrix[row + (column + i) * rows] * work[i +
            threadIdx.y * 2 * fastEIT::matrix::block_size] :
            0.0f;
    }

    // set result
    if (row < rows) {
        result[row + (threadIdx.y + blockIdx.y * blockDim.y) * rows] = product;
    }
}

// gemv kernel wrapper
void fastEIT::numeric::conjugateKernel::gemv(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dtype::real* matrix, const dtype::real* vector,
    dtype::size rows, dtype::real* result) {
    // call cuda kernel
    gemvKernel<<<blocks, threads, 0, stream>>>(matrix, vector, rows, result);
}

// row reduce kernel
static __global__ void reduceRowKernel(fastEIT::dtype::size rows,
    fastEIT::dtype::real* vector) {
    // get id
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // check row
    if (row >= rows) {
        return;
    }

    // sum row
    fastEIT::dtype::real sum = 0.0f;
    fastEIT::dtype::size count =
        (rows + 2 * fastEIT::matrix::block_size - 1) /
        (2 * fastEIT::matrix::block_size);
    for (fastEIT::dtype::index i = 0; i < count; i++) {
        sum += vector[row + i * rows];
    }

    // set sum
    vector[row] = sum;
}

// row reduce kernel wrapper
void fastEIT::numeric::conjugateKernel::reduceRow(dim3 blocks, dim3 threads,
    cudaStream_t stream, dtype::size rows, dtype::real* vector) {
    // call cuda kernel
    reduceRowKernel<<<blocks, threads, 0, stream>>>(rows, vector);
}
