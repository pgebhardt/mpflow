// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// add scalar kernel
__global__ void add_scalar_kernel(linalgcu_matrix_data_t* vector,
    linalgcu_matrix_data_t* scalar, linalgcu_size_t size) {
    // get id
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // add data
    vector[i] += i < size ? scalar[0] : 0.0f;
}

// add scalar
extern "C"
linalgcu_error_t fastect_conjugate_add_scalar(linalgcu_matrix_t vector,
    linalgcu_matrix_t scalar, linalgcu_size_t size, cudaStream_t stream) {
    // check input
    if ((vector == NULL) || (scalar == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    add_scalar_kernel<<<vector->size_m / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE,
        0, stream>>>(vector->device_data, scalar->device_data, size);

    return LINALGCU_SUCCESS;
}

// update vector
__global__ void update_vector_kernel(linalgcu_matrix_data_t* result,
    linalgcu_matrix_data_t* x1, linalgcu_matrix_data_t sign,
    linalgcu_matrix_data_t* x2, linalgcu_matrix_data_t* r1, linalgcu_matrix_data_t* r2) {
    // get id
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // calc value
    result[i] = x1[i] + sign * x2[i] * r1[0] / r2[0];
}

// update vector
extern "C"
linalgcu_error_t fastect_conjugate_udate_vector(linalgcu_matrix_t result,
    linalgcu_matrix_t x1, linalgcu_matrix_data_t sign, linalgcu_matrix_t x2,
    linalgcu_matrix_t r1, linalgcu_matrix_t r2, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (x1 == NULL) || (x2 == NULL) || (r1 == NULL) || (r2 == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    update_vector_kernel<<<result->size_m / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE,
        0, stream>>>(result->device_data, x1->device_data, sign, x2->device_data,
        r1->device_data, r2->device_data);

    return LINALGCU_SUCCESS;
}

// gemv kernel
__global__ void gemv_kernel(linalgcu_matrix_data_t* A, linalgcu_matrix_data_t* x,
    linalgcu_matrix_data_t* y, linalgcu_size_t size_m) {
    // column
    linalgcu_size_t col0 = blockIdx.y * LINALGCU_BLOCK_SIZE;

    // Load one slice of x in work
    __shared__ linalgcu_matrix_data_t work[LINALGCU_BLOCK_SIZE];
    work[threadIdx.x] = x[col0 + threadIdx.x];
    __syncthreads();

    // compute partial dot product
    linalgcu_matrix_data_t sum = 0.0f;
    for (linalgcu_size_t k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        sum += A[(blockIdx.x * blockDim.x + threadIdx.x) + (col0 + k) * size_m] * work[k];
    }

    // store to y
    y[(blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * size_m] = sum;
}

// row reduce kernel
__global__ void reduce_row_kernel(linalgcu_matrix_data_t* vector, linalgcu_size_t size_m) {
    // get id
    linalgcu_size_t column = blockIdx.x * blockDim.x + threadIdx.x;

    // sum row
    linalgcu_matrix_data_t sum = 0.0f;
    for (linalgcu_size_t i = 0; i < size_m / LINALGCU_BLOCK_SIZE; i++) {
        sum += vector[column + i * size_m];
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
    dim3 blocks(A->size_m / LINALGCU_BLOCK_SIZE, A->size_n / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, 1);

    // call gemv kernel
    gemv_kernel<<<blocks, threads, 0, stream>>>(A->device_data, x->device_data,
        y->device_data, A->size_m);

    // call reduce kernel
    reduce_row_kernel<<<A->size_n / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE, 0, stream>>>(
        y->device_data, y->size_m);

    return LINALGCU_SUCCESS;
}

