// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/cuda_error.h"

#include "../include/dtype.h"
#include "../include/constants.h"
#include "../include/model_kernel.h"

// reduce connectivity and elementalResidual matrix
template <
    class type
>
static __global__ void reduceMatrixKernel(const type* intermediateMatrix,
    const fastEIT::dtype::index* systemMatrixColumnIds, fastEIT::dtype::size rows,
    type* matrix) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    fastEIT::dtype::index columnId = systemMatrixColumnIds[row * fastEIT::sparseMatrix::block_size + column];

    // check column id
    if (columnId == fastEIT::dtype::invalid_index) {
        return;
    }

    // reduce matrices
    for (fastEIT::dtype::index k = 0; k < fastEIT::matrix::block_size; ++k) {
        matrix[row + (column + k * fastEIT::sparseMatrix::block_size) * rows] =
            intermediateMatrix[row + (columnId + k * rows) * rows];
    }
}

// reduce matrix wrapper
template <
    class type
>
void fastEIT::modelKernel::reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* intermediateMatrix, const dtype::index* systemMatrixColumnIds,
    dtype::size rows, type* matrix) {
    // call cuda kernel
    reduceMatrixKernel<type><<<blocks, threads, 0, stream>>>(intermediateMatrix,
        systemMatrixColumnIds, rows, matrix);

    CudaCheckError();
}

// reduce matrix specialisation
template void fastEIT::modelKernel::reduceMatrix<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::modelKernel::reduceMatrix<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::index*);

// update matrix kernel
static __global__ void updateMatrixKernel(const fastEIT::dtype::index* connectivityMatrix,
    const fastEIT::dtype::real* elementalMatrix, const fastEIT::dtype::real* gamma,
    fastEIT::dtype::real sigmaRef, fastEIT::dtype::size rows,
    fastEIT::dtype::real* matrix_values) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    fastEIT::dtype::real value = 0.0f;
    fastEIT::dtype::index elementId = fastEIT::dtype::invalid_index;
    for (fastEIT::dtype::index k = 0; k < fastEIT::matrix::block_size; ++k) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * fastEIT::sparseMatrix::block_size) * rows];

        value += elementId != fastEIT::dtype::invalid_index ? elementalMatrix[row +
            (column + k * fastEIT::sparseMatrix::block_size) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrix_values[row * fastEIT::sparseMatrix::block_size + column] = value;
}

// update matrix kernel wrapper
void fastEIT::modelKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::index* connectivityMatrix, const dtype::real* elementalMatrix,
    const dtype::real* gamma, dtype::real sigma_ref, dtype::size rows,
    dtype::real* matrix_values) {
    // call cuda kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(connectivityMatrix, elementalMatrix,
        gamma, sigma_ref, rows, matrix_values);

    CudaCheckError();
}
