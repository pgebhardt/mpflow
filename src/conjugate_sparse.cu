// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// add scalar kernel
__global__ void add_scalar_kernel(linalgcuMatrixData_t* vector,
    linalgcuMatrixData_t* scalar, linalgcuSize_t vector_rows,
    linalgcuSize_t rows, linalgcuSize_t columns) {
    // get row
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // get column
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vector_rows] += row < rows && column < columns ? scalar[column * vector_rows] : 0.0f;
}

// add scalar
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_sparse_add_scalar(linalgcuMatrix_t vector,
    linalgcuMatrix_t scalar, linalgcuSize_t rows, linalgcuSize_t columns, cudaStream_t stream) {
    // check input
    if ((vector == NULL) || (scalar == NULL)) {
        return LINALGCU_ERROR;
    }

    // kernel dimension
    dim3 global(vector->rows / LINALGCU_BLOCK_SIZE, vector->columns / LINALGCU_BLOCK_SIZE);
    dim3 local(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // execute kernel
    add_scalar_kernel<<<global, local, 0, stream>>>(vector->deviceData, scalar->deviceData,
        vector->rows, rows, columns);

    return LINALGCU_SUCCESS;
}

// update vector
__global__ void sparse_update_vector_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* x1, linalgcuMatrixData_t sign,
    linalgcuMatrixData_t* x2, linalgcuMatrixData_t* r1, linalgcuMatrixData_t* r2,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] + sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_sparse_update_vector(linalgcuMatrix_t result,
    linalgcuMatrix_t x1, linalgcuMatrixData_t sign, linalgcuMatrix_t x2,
    linalgcuMatrix_t r1, linalgcuMatrix_t r2, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (x1 == NULL) || (x2 == NULL) || (r1 == NULL) || (r2 == NULL)) {
        return LINALGCU_ERROR;
    }

    // kernel dimension
    dim3 global(result->rows / LINALGCU_BLOCK_SIZE, result->columns / LINALGCU_BLOCK_SIZE);
    dim3 local(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // execute kernel
    sparse_update_vector_kernel<<<global, local, 0, stream>>>(result->deviceData,
        x1->deviceData, sign, x2->deviceData, r1->deviceData, r2->deviceData, result->rows);

    return LINALGCU_SUCCESS;
}
