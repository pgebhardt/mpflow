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
__global__ void gemv_kernel(linalgcuMatrixData_t* matrix, linalgcuMatrixData_t* vector,
    linalgcuMatrixData_t* result, linalgcuSize_t rows) {
    // column
    linalgcuSize_t j = blockIdx.y * blockDim.y + threadIdx.y;
    linalgcuSize_t col0 = j * LINALGCU_BLOCK_SIZE;

    // Load one slice of vector in work
    __shared__ linalgcuMatrixData_t work[LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE];
    work[threadIdx.x + threadIdx.y * LINALGCU_BLOCK_SIZE] = col0 + threadIdx.x < rows ?
        vector[col0 + threadIdx.x] : 0.0f;
    __syncthreads();

    // compute partial dot product
    linalgcuMatrixData_t sum = 0.0f;
    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        sum += col0 + k < rows ? matrix[
            (blockIdx.x * blockDim.x + threadIdx.x) + (col0 + k) * rows] *
            work[k + threadIdx.y * LINALGCU_BLOCK_SIZE] : 0.0f;
    }

    // store to result
    if (j < rows / LINALGCU_BLOCK_SIZE) {
        result[(blockIdx.x * blockDim.x + threadIdx.x) + j * rows] = sum;
    }
}

// row reduce kernel
__global__ void reduce_row_kernel(linalgcuMatrixData_t* vector, linalgcuSize_t rows) {
    // get id
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // check row
    if (row >= rows) {
        return;
    }

    // sum row
    linalgcuMatrixData_t sum = 0.0f;
    for (int i = 0; i < rows / LINALGCU_BLOCK_SIZE; i++) {
        sum += vector[row + i * rows];
    }

    // set sum
    vector[row] = sum;
}

// fast gemv
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_gemv(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
    linalgcuMatrix_t vector, cudaStream_t stream) {
    // check input
    if ((matrix == NULL) || (vector == NULL) || (result == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE,
        (matrix->columns / LINALGCU_BLOCK_SIZE + LINALGCU_BLOCK_SIZE - 1) / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // call gemv kernel
    gemv_kernel<<<blocks, threads, 0, stream>>>(matrix->deviceData, vector->deviceData,
        result->deviceData, matrix->rows);

    // call reduce kernel
    reduce_row_kernel<<<(matrix->columns + LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE - 1) /
        (LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE), LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE,
        0, stream>>>(result->deviceData, result->rows);

    return LINALGCU_SUCCESS;
}

