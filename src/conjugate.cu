// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// update vector
__global__ void update_vector_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* x1, linalgcuMatrixData_t sign,
    linalgcuMatrixData_t* x2, linalgcuMatrixData_t* r1, linalgcuMatrixData_t* r2) {
    // get id
    linalgcuSize_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // calc value
    result[i] = r2[0] != 0.0f ? x1[i] + sign * x2[i] * r1[0] / r2[0] : 0.0f;
}

// update vector
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_update_vector(linalgcuMatrix_t result,
    linalgcuMatrix_t x1, linalgcuMatrixData_t sign, linalgcuMatrix_t x2,
    linalgcuMatrix_t r1, linalgcuMatrix_t r2, cudaStream_t stream) {
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
__global__ void gemv_kernel(linalgcuMatrixData_t* A, linalgcuMatrixData_t* x,
    linalgcuMatrixData_t* y, linalgcuSize_t rows) {
    // column
    linalgcuSize_t col0 = blockIdx.y * LINALGCU_BLOCK_SIZE;

    // Load one slice of x in work
    __shared__ linalgcuMatrixData_t work[LINALGCU_BLOCK_SIZE];
    work[threadIdx.x] = x[col0 + threadIdx.x];
    __syncthreads();

    // compute partial dot product
    linalgcuMatrixData_t sum = 0.0f;
    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        sum += A[(blockIdx.x * blockDim.x + threadIdx.x) + (col0 + k) * rows] * work[k];
    }

    // store to y
    y[(blockIdx.x * blockDim.x + threadIdx.x) + blockIdx.y * rows] = sum;
}

// row reduce kernel
__global__ void reduce_row_kernel(linalgcuMatrixData_t* vector, linalgcuSize_t rows) {
    // get id
    linalgcuSize_t column = blockIdx.x * blockDim.x + threadIdx.x;

    // sum row
    linalgcuMatrixData_t sum = 0.0f;
    for (int i = 0; i < rows / LINALGCU_BLOCK_SIZE; i++) {
        sum += vector[column + i * rows];
    }

    // set sum
    vector[column] = sum;
}

// fast gemv
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_gemv(linalgcuMatrix_t A, linalgcuMatrix_t x,
    linalgcuMatrix_t y, cudaStream_t stream) {
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

