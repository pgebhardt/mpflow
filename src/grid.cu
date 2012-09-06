// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// update_system_matrix_kernel
__global__ void update_system_matrix_kernel(linalgcu_matrix_data_t* systemMatrixValues,
    linalgcu_column_id_t* systemMatrixColumnIds,
    linalgcu_matrix_data_t* gradientMatrixTransposedValues,
    linalgcu_column_id_t* gradientMatrixTransposedColumnIds,
    linalgcu_matrix_data_t* gradientMatrixTransposed,
    linalgcu_matrix_data_t* sigma,
    linalgcu_matrix_data_t* area,
    linalgcu_size_t gradientMatrixTransposedRows) {
    // get ids
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcu_column_id_t j = systemMatrixColumnIds[i * LINALGCU_BLOCK_SIZE +
        (blockIdx.y * blockDim.y + threadIdx.y)];

    // calc system matrix elements
    linalgcu_matrix_data_t element = 0.0f;
    linalgcu_column_id_t id = -1;

    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        // get id
        id = gradientMatrixTransposedColumnIds[i * LINALGCU_BLOCK_SIZE + k];

        element += id != -1 && j != -1 ?
            gradientMatrixTransposedValues[i * LINALGCU_BLOCK_SIZE + k] *
            sigma[id / 2] * area[id / 2] *
            gradientMatrixTransposed[j + id * gradientMatrixTransposedRows] :
            0.0f;
    }

    // set element
    systemMatrixValues[i * LINALGCU_BLOCK_SIZE + (blockIdx.y * blockDim.y + threadIdx.y)] =
        element;
}

// update system matrix
extern "C"
linalgcu_error_t fastect_grid_update_system_matrix(fastect_grid_t grid,
    linalgcu_matrix_t sigma, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (sigma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(grid->systemMatrix->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_system_matrix_kernel<<<blocks, threads, 0, stream>>>(
        grid->systemMatrix->values,
        grid->systemMatrix->columnIds,
        grid->gradientMatrixTransposedSparse->values,
        grid->gradientMatrixTransposedSparse->columnIds,
        grid->gradientMatrixTransposed->deviceData,
        sigma->deviceData,
        grid->area->deviceData,
        grid->gradientMatrixTransposed->rows);

    return LINALGCU_SUCCESS;
}

// update residual matrix kernel
__global__ void update_residual_matrix_kernel(linalgcu_matrix_data_t* residualMatrixValues,
    linalgcu_column_id_t* residualMatrixColumnIds,
    linalgcu_column_id_t* gradientMatrixTransposedColumnIds,
    linalgcu_matrix_data_t* gradientMatrixTransposed,
    linalgcu_matrix_data_t* integralMatrix, linalgcu_matrix_data_t* sigma,
    linalgcu_size_t gradientMatrixTransposedRows) {
    // get ids
    linalgcu_size_t nodeI = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcu_column_id_t nodeJ = residualMatrixColumnIds[nodeI * LINALGCU_BLOCK_SIZE +
        blockIdx.y * blockDim.y + threadIdx.y];

    // calc residual matrix values
    linalgcu_matrix_data_t value = 0.0f;
    linalgcu_column_id_t element = -1;

    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        // get element
        element = gradientMatrixTransposedColumnIds[nodeI * LINALGCU_BLOCK_SIZE + k];

    }

    // set value
    residualMatrixValues[nodeI * LINALGCU_BLOCK_SIZE + (blockIdx.y * blockDim.y + threadIdx.y)] =
        value;
}

// update residual matrix
extern "C"
linalgcu_error_t fastect_grid_update_residual_matrix(fastect_grid_t grid,
    linalgcu_matrix_t sigma, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (sigma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(grid->residualMatrix->rows / LINALGCU_BLOCK_SIZE, 1);

    return LINALGCU_SUCCESS;
}
