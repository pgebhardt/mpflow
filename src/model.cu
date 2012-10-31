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
__global__ void reduce_residual_matrices(linalgcuMatrixData_t* connectivityMatrix,
    linalgcuMatrixData_t* elementalResidualMatrix,
    linalgcuMatrixData_t* intermediateConnectivityMatrix,
    linalgcuMatrixData_t* intermediateElementalResidualMatrix,
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
        connectivityMatrix[row + (column + k * LINALGCU_BLOCK_SIZE) * rows] =
            intermediateConnectivityMatrix[row + (columnId + k * columns) * rows];

        elementalResidualMatrix[row + (column + k * LINALGCU_BLOCK_SIZE) * rows] =
            intermediateElementalResidualMatrix[row + (columnId + k * columns) * rows];
    }
}

// init residual matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_init_residual_matrix(fasteitModel_t self,
    linalgcuMatrix_t gamma, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // create intermediate matrices
    linalgcuMatrix_t elementCount, connectivityMatrix, elementalResidualMatrix;
    error  = linalgcu_matrix_create(&elementCount, self->mesh->vertexCount,
        self->mesh->vertexCount, stream);
    error |= linalgcu_matrix_create(&connectivityMatrix, self->connectivityMatrix->rows,
        elementCount->columns * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&elementalResidualMatrix,
        self->elementalResidualMatrix->rows, elementCount->columns * LINALGCU_BLOCK_SIZE,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // init connectivityMatrix
    for (linalgcuSize_t i = 0; i < self->connectivityMatrix->rows; i++) {
        for (linalgcuSize_t j = 0; j < self->connectivityMatrix->columns; j++) {
            linalgcu_matrix_set_element(self->connectivityMatrix, -1.0f, i, j);
        }
    }
    linalgcu_matrix_copy_to_device(self->connectivityMatrix, stream);

    // fill intermediate connectivity and elementalResidual matrices
    linalgcuMatrixData_t id[3], x[3], y[3];
    linalgcuMatrixData_t temp;
    fasteitBasis_t basis[3];

    for (linalgcuSize_t k = 0; k < self->mesh->elementCount; k++) {
        // get vertices for element
        for (linalgcuSize_t i = 0; i < 3; i++) {
            linalgcu_matrix_get_element(self->mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(self->mesh->vertices, &x[i],
                (linalgcuSize_t)id[i], 0);
            linalgcu_matrix_get_element(self->mesh->vertices, &y[i],
                (linalgcuSize_t)id[i], 1);
        }

        // calc corresponding basis functions
        fasteit_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        fasteit_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        fasteit_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // set connectivity and elemental residual matrix elements
        for (linalgcuSize_t i = 0; i < 3; i++) {
            for (linalgcuSize_t j = 0; j < 3; j++) {
                // get current element count
                linalgcu_matrix_get_element(elementCount, &temp,
                    (linalgcuSize_t)id[i], (linalgcuSize_t)id[j]);

                // set connectivity element
                linalgcu_matrix_set_element(connectivityMatrix,
                    (linalgcuMatrixData_t)k, (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // set elemental residual element
                linalgcu_matrix_set_element(elementalResidualMatrix,
                    fasteit_basis_integrate_with_basis(basis[i], basis[j]),
                    (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // increment element count
                elementCount->hostData[(linalgcuSize_t)id[i] + (linalgcuSize_t)id[j] *
                    elementCount->rows] += 1.0f;
            }
        }

        // cleanup
        fasteit_basis_release(&basis[0]);
        fasteit_basis_release(&basis[1]);
        fasteit_basis_release(&basis[2]);
    }

    // upload intermediate matrices
    linalgcu_matrix_copy_to_device(connectivityMatrix, stream);
    linalgcu_matrix_copy_to_device(elementalResidualMatrix, stream);

    // reduce matrices
    dim3 blocks(connectivityMatrix->rows / LINALGCU_BLOCK_SIZE, 1);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    reduce_residual_matrices<<<blocks, threads, 0, stream>>>(
        self->connectivityMatrix->deviceData,
        self->elementalResidualMatrix->deviceData,
        connectivityMatrix->deviceData,
        elementalResidualMatrix->deviceData,
        self->systemMatrix2D->columnIds,
        self->connectivityMatrix->rows, self->connectivityMatrix->rows);

    // update residual matrix
    error = fasteit_model_update_residual_matrix(self, gamma, stream);

    // cleanup
    linalgcu_matrix_release(&elementCount);
    linalgcu_matrix_release(&connectivityMatrix);
    linalgcu_matrix_release(&elementalResidualMatrix);

    return LINALGCU_SUCCESS;
}

// update_system_matrix_kernel
__global__ void update_system_matrix_kernel(linalgcuMatrixData_t* systemMatrixValues,
    linalgcuColumnId_t* systemMatrixColumnIds,
    linalgcuMatrixData_t* gradientMatrixTransposedValues,
    linalgcuColumnId_t* gradientMatrixTransposedColumnIds,
    linalgcuMatrixData_t* gradientMatrixTransposed,
    linalgcuMatrixData_t* gamma, linalgcuMatrixData_t sigmaRef,
    linalgcuMatrixData_t* area,
    linalgcuSize_t gradientMatrixTransposedRows) {
    // get ids
    linalgcuSize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuColumnId_t j = systemMatrixColumnIds[i * LINALGCU_BLOCK_SIZE +
        (blockIdx.y * blockDim.y + threadIdx.y)];

    // calc system matrix elements
    linalgcuMatrixData_t element = 0.0f;
    linalgcuColumnId_t id = -1;

    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        // get id
        id = gradientMatrixTransposedColumnIds[i * LINALGCU_BLOCK_SIZE + k];

        element += id != -1 && j != -1 ?
            gradientMatrixTransposedValues[i * LINALGCU_BLOCK_SIZE + k] *
            sigmaRef * exp10f(gamma[id / 2] / 10.0f) * area[id / 2] *
            gradientMatrixTransposed[j + id * gradientMatrixTransposedRows] :
            0.0f;
    }

    // set element
    systemMatrixValues[i * LINALGCU_BLOCK_SIZE + (blockIdx.y * blockDim.y + threadIdx.y)] =
        element;
}

// update system matrix 2D
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_update_2D_system_matrix(fasteitModel_t self,
    linalgcuMatrix_t gamma, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(self->systemMatrix2D->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_system_matrix_kernel<<<blocks, threads, 0, stream>>>(
        self->systemMatrix2D->values,
        self->systemMatrix2D->columnIds,
        self->gradientMatrixTransposedSparse->values,
        self->gradientMatrixTransposedSparse->columnIds,
        self->gradientMatrixTransposed->deviceData,
        gamma->deviceData, self->sigmaRef,
        self->area->deviceData,
        self->gradientMatrixTransposed->rows);

    return LINALGCU_SUCCESS;
}

// update residual matrix kernel
__global__ void update_residual_matrix_kernel(linalgcuMatrixData_t* residualMatrixValues,
    linalgcuColumnId_t* residualMatrixColumnIds,
    linalgcuColumnId_t* systemMatrixColumnIds,
    linalgcuMatrixData_t* connectivityMatrix,
    linalgcuMatrixData_t* elementalResidualMatrix,
    linalgcuMatrixData_t* gamma, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get columnId
    linalgcuColumnId_t columnId = systemMatrixColumnIds[row * LINALGCU_BLOCK_SIZE + column];

    // set column id
    residualMatrixColumnIds[row * LINALGCU_BLOCK_SIZE + column] = columnId;

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

        value += elementId != -1 ? elementalResidualMatrix[row +
            (column + k * LINALGCU_BLOCK_SIZE) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    residualMatrixValues[row * LINALGCU_BLOCK_SIZE + column] = value;
}

// update residual matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_update_residual_matrix(fasteitModel_t self,
    linalgcuMatrix_t gamma, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(self->residualMatrix->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_residual_matrix_kernel<<<blocks, threads, 0, stream>>>(
        self->residualMatrix->values, self->residualMatrix->columnIds,
        self->systemMatrix2D->columnIds, self->connectivityMatrix->deviceData,
        self->elementalResidualMatrix->deviceData, gamma->deviceData,
        self->sigmaRef, self->connectivityMatrix->rows);

    return LINALGCU_SUCCESS;
}
