// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// update vector
__global__ void update_vector_kernel(linalgcu_matrix_data_t* result,
    linalgcu_matrix_data_t* x1, linalgcu_matrix_data_t sign,
    linalgcu_matrix_data_t* x2, linalgcu_matrix_data_t* r1, linalgcu_matrix_data_t* r2) {
    // get id
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // calc value
    result[i] = r2[0] != 0.0f ? x1[i] + sign * x2[i] * r1[0] / r2[0] : 0.0f;
}

// update vector
extern "C"
linalgcu_error_t fastect_conjugate_update_vector(linalgcu_matrix_t result,
    linalgcu_matrix_t x1, linalgcu_matrix_data_t sign, linalgcu_matrix_t x2,
    linalgcu_matrix_t r1, linalgcu_matrix_t r2, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (x1 == NULL) || (x2 == NULL) || (r1 == NULL) || (r2 == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    update_vector_kernel<<<result->rows / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE,
        0, stream>>>(result->deviceData, x1->deviceData, sign, x2->deviceData,
        r1->deviceData, r2->deviceData);

    return LINALGCU_SUCCESS;
}

// gemv kernel
__global__ void gemv_kernel(linalgcu_matrix_data_t* A, linalgcu_matrix_data_t* x,
    linalgcu_matrix_data_t* y, linalgcu_size_t rows) {
    // column
    linalgcu_size_t col0 = blockIdx.y * LINALGCU_BLOCK_SIZE;

    // Load one slice of x in work
    __shared__ linalgcu_matrix_data_t work[LINALGCU_BLOCK_SIZE];
    work[threadIdx.x] = x[col0 + threadIdx.x];
    __syncthreads();

    // compute partial dot product
    linalgcu_matrix_data_t sum = 0.0f;
    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        sum += A[(blockIdx.x * blockDim.x + threadIdx.x) + (col0 + k) * rows] * work[k];
    }

    // store to y
    y[(blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * rows] = sum;
}

// row reduce kernel
__global__ void reduce_row_kernel(linalgcu_matrix_data_t* vector, linalgcu_size_t rows) {
    // get id
    linalgcu_size_t column = blockIdx.x * blockDim.x + threadIdx.x;

    // sum row
    linalgcu_matrix_data_t sum = 0.0f;
    for (int i = 0; i < rows / LINALGCU_BLOCK_SIZE; i++) {
        sum += vector[column + i * rows];
    }

    // set sum
    vector[column] = sum;
}

// fast gemv
extern "C"
linalgcu_error_t fastect_conjugate_gemv(linalgcu_matrix_t A, linalgcu_matrix_t x,
    linalgcu_matrix_t y, cudaStream_t stream) {
    // check input
    if ((A == NULL) || (x == NULL) || (y == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(A->rows / LINALGCU_BLOCK_SIZE, A->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, 1);

    // call gemv kernel
    gemv_kernel<<<blocks, threads, 0, stream>>>(A->deviceData, x->deviceData,
        y->deviceData, A->rows);

    // call reduce kernel
    reduce_row_kernel<<<A->columns / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE, 0, stream>>>(
        y->deviceData, y->rows);

    return LINALGCU_SUCCESS;
}

