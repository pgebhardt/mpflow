// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include <stdio.h>
#include "../include/fasteit.h"

// reduce connectivity and elementalResidual matrix
__global__ void reduce_matrix_kernel(linalgcuMatrixData_t* matrix,
    linalgcuMatrixData_t* intermediateMatrix,
    linalgcuColumnId_t* systemMatrixColumnIds, linalgcuSize_t rows,
    linalgcuSize_t columns) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    linalgcuColumnId_t columnId = systemMatrixColumnIds[row * LINALGCU_BLOCK_SIZE + column];

    // check column id
    if (columnId == -1) {
        return;
    }

    // reduce matrices
    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        matrix[row + (column + k * LINALGCU_BLOCK_SIZE) * rows] =
            intermediateMatrix[row + (columnId + k * columns) * rows];
    }
}

// reduce matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_reduce_matrix(fasteitModel_t self, linalgcuMatrix_t matrix,
    linalgcuMatrix_t intermediateMatrix, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (matrix == NULL) || (intermediateMatrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // block size
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE, 1);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // reduce matrix
    reduce_matrix_kernel<<<blocks, threads, 0, stream>>>(
        matrix->deviceData, intermediateMatrix->deviceData,
        self->systemMatrix2D->columnIds, matrix->rows, matrix->rows);

    return LINALGCU_SUCCESS;
}

// update matrix kernel
__global__ void update_matrix_kernel(linalgcuMatrixData_t* matrixValues,
    linalgcuColumnId_t* matrixColumnIds, linalgcuColumnId_t* columnIds,
    linalgcuMatrixData_t* connectivityMatrix, linalgcuMatrixData_t* elementalMatrix,
    linalgcuMatrixData_t* gamma, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get columnId
    linalgcuColumnId_t columnId = columnIds[row * LINALGCU_BLOCK_SIZE + column];

    // set column id
    matrixColumnIds[row * LINALGCU_BLOCK_SIZE + column] = columnId;

    // check column id
    if (columnId == -1) {
        return;
    }

    // calc residual matrix element
    linalgcuMatrixData_t value = 0.0f;
    linalgcuColumnId_t elementId = -1;
    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        // get element id
        elementId = (linalgcuColumnId_t)connectivityMatrix[row +
            (column + k * LINALGCU_BLOCK_SIZE) * rows];

        value += elementId != -1 ? elementalMatrix[row +
            (column + k * LINALGCU_BLOCK_SIZE) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrixValues[row * LINALGCU_BLOCK_SIZE + column] = value;
}

// update matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_update_matrix(fasteitModel_t self,
    linalgcuSparseMatrix_t matrix, linalgcuMatrix_t elementalMatrix, linalgcuMatrix_t gamma,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (matrix == NULL) || (elementalMatrix == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_matrix_kernel<<<blocks, threads, 0, stream>>>(
        matrix->values, matrix->columnIds, self->systemMatrix2D->columnIds,
        self->connectivityMatrix->deviceData, elementalMatrix->deviceData,
        gamma->deviceData, self->sigmaRef, self->connectivityMatrix->rows);

    return LINALGCU_SUCCESS;
}
