// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/numeric/conjugate_kernel.h"

// add scalar kernel
static __global__ void addScalarKernel(const mpFlow::dtype::real* scalar,
    mpFlow::dtype::size vectorRows, mpFlow::dtype::size rows,
    mpFlow::dtype::size columns, mpFlow::dtype::real* vector) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vectorRows] += row < rows && column < columns ?
        scalar[column * vectorRows] : 0.0f;
}

// add scalar kernel wrapper
void mpFlow::numeric::conjugateKernel::addScalar(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dtype::real* scalar, dtype::size vector_rows,
    dtype::size rows, dtype::size columns, dtype::real* vector) {
    // call cuda kernel
    addScalarKernel<<<blocks, threads, 0, stream>>>(scalar, vector_rows,
        rows, columns, vector);

    CudaCheckError();
}

// update vector kernel
static __global__ void updateVectorKernel(const mpFlow::dtype::real* x1,
    const mpFlow::dtype::real sign, const mpFlow::dtype::real* x2,
    const mpFlow::dtype::real* r1, const mpFlow::dtype::real* r2,
    mpFlow::dtype::size rows, mpFlow::dtype::real* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] +
        sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector kernel wrapper
void mpFlow::numeric::conjugateKernel::updateVector(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dtype::real* x1, const dtype::real sign,
    const dtype::real* x2, const dtype::real* r1, const dtype::real* r2,
    dtype::size rows, dtype::real* result) {
    // call cuda kernel
    updateVectorKernel<<<blocks, threads, 0, stream>>>(x1, sign, x2, r1, r2, rows,
        result);

    CudaCheckError();
}

// gemv kernel
static __global__ void gemvKernel(const mpFlow::dtype::real* matrix,
    const mpFlow::dtype::real* vector, mpFlow::dtype::size rows,
    mpFlow::dtype::real* result) {
    // get ids
    mpFlow::dtype::index row = threadIdx.x + blockIdx.x * blockDim.x;
    mpFlow::dtype::index column = (threadIdx.y + blockIdx.y * blockDim.y) *
        2 * mpFlow::numeric::matrix::block_size;

    // load vector to shared memory
    __shared__ mpFlow::dtype::real work[2 * mpFlow::numeric::matrix::block_size *
        mpFlow::numeric::matrix::block_size];
    work[threadIdx.x +
        threadIdx.y * 2 * mpFlow::numeric::matrix::block_size] =
        column + threadIdx.x < rows ? vector[column + threadIdx.x] : 0.0f;
    __syncthreads();

    // compute partial vector product
    mpFlow::dtype::real product = 0.0f;
    for (mpFlow::dtype::index i = 0; i < 2 * mpFlow::numeric::matrix::block_size; i++) {
        product += row < rows && column + i < rows ?
            matrix[row + (column + i) * rows] * work[i +
            threadIdx.y * 2 * mpFlow::numeric::matrix::block_size] :
            0.0f;
    }

    // set result
    if (row < rows) {
        result[row + (threadIdx.y + blockIdx.y * blockDim.y) * rows] = product;
    }
}

// gemv kernel wrapper
void mpFlow::numeric::conjugateKernel::gemv(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dtype::real* matrix, const dtype::real* vector,
    dtype::size rows, dtype::real* result) {
    // call cuda kernel
    gemvKernel<<<blocks, threads, 0, stream>>>(matrix, vector, rows, result);

    CudaCheckError();
}

// row reduce kernel
static __global__ void reduceRowKernel(mpFlow::dtype::size rows,
    mpFlow::dtype::real* vector) {
    // get id
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // check row
    if (row >= rows) {
        return;
    }

    // sum row
    mpFlow::dtype::real sum = 0.0f;
    mpFlow::dtype::size count =
        (rows + 2 * mpFlow::numeric::matrix::block_size - 1) /
        (2 * mpFlow::numeric::matrix::block_size);
    for (mpFlow::dtype::index i = 0; i < count; i++) {
        sum += vector[row + i * rows];
    }

    // set sum
    vector[row] = sum;
}

// row reduce kernel wrapper
void mpFlow::numeric::conjugateKernel::reduceRow(dim3 blocks, dim3 threads,
    cudaStream_t stream, dtype::size rows, dtype::real* vector) {
    // call cuda kernel
    reduceRowKernel<<<blocks, threads, 0, stream>>>(rows, vector);

    CudaCheckError();
}
