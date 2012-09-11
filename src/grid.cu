// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// calc residual integral
linalgcu_matrix_data_t calc_residual_integral(
    linalgcu_matrix_data_t x1, linalgcu_matrix_data_t y1,
    linalgcu_matrix_data_t x2, linalgcu_matrix_data_t y2,
    linalgcu_matrix_data_t x3, linalgcu_matrix_data_t y3,
    linalgcu_matrix_data_t ai, linalgcu_matrix_data_t bi, linalgcu_matrix_data_t ci,
    linalgcu_matrix_data_t aj, linalgcu_matrix_data_t bj, linalgcu_matrix_data_t cj) {
    // calc area
    linalgcu_matrix_data_t area = 0.5 * fabs((x2 - x1) * (y3 - y1) -
        (x3 - x1) * (y2 - y1));

    // calc integral
    linalgcu_matrix_data_t integral = 2.0f * area *
        (ai * (0.5f * aj + (1.0f / 6.0f) * bj * (x1 + x2 + x3) +
        (1.0f / 6.0f) * cj * (y1 + y2 + y3)) +
        bi * ((1.0f/ 6.0f) * aj * (x1 + x2 + x3) +
        (1.0f / 12.0f) * bj * (x1 * x1 + x1 * x2 + x1 * x3 + x2 * x2 + x2 * x3 + x3 * x3) +
        (1.0f/ 24.0f) * cj * (2.0f * x1 * y1 + x1 * y2 + x1 * y3 + x2 * y1 +
        2.0f * x2 * y2 + x2 * y3 + x3 * y1 + x3 * y2 + 2.0f * x3 * y3)) +
        ci * ((1.0f / 6.0f) * aj * (y1 + y2 + y3) +
        (1.0f / 12.0f) * cj * (y1 * y1 + y1 * y2 + y1 * y3 + y2 * y2 + y2 * y3 + y3 * y3) +
        (1.0f / 24.0f) * bj * (2.0f * x1 * y1 + x1 * y2 + x1 * y3 + x2 * y1 +
        2.0f * x2 * y2 + x2 * y3 + x3 * y1 + x3 * y2 + 2.0f * x3 * y3)));

    return integral;
}

// reduce connectivity and elementalResidual matrix
__global__ void reduce_residual_matrices(linalgcu_matrix_data_t* connectivityMatrix,
    linalgcu_matrix_data_t* elementalResidualMatrix,
    linalgcu_matrix_data_t* intermediateConnectivityMatrix,
    linalgcu_matrix_data_t* intermediateElementalResidualMatrix,
    linalgcu_column_id_t* systemMatrixColumnIds, linalgcu_size_t rows,
    linalgcu_size_t columns) {
    // get ids
    linalgcu_size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcu_size_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    linalgcu_column_id_t columnId = systemMatrixColumnIds[row * LINALGCU_BLOCK_SIZE + column];

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
linalgcu_error_t fastect_grid_init_residual_matrix(fastect_grid_t grid,
    linalgcu_matrix_t gamma, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // create intermediate matrices
    linalgcu_matrix_t elementCount, connectivityMatrix, elementalResidualMatrix;
    error  = linalgcu_matrix_create(&elementCount, grid->mesh->vertexCount,
        grid->mesh->vertexCount, stream);
    error |= linalgcu_matrix_create(&connectivityMatrix, grid->connectivityMatrix->rows,
        elementCount->columns * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&elementalResidualMatrix,
        grid->elementalResidualMatrix->rows, elementCount->columns * LINALGCU_BLOCK_SIZE,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // init connectivityMatrix
    for (linalgcu_size_t i = 0; i < grid->connectivityMatrix->rows; i++) {
        for (linalgcu_size_t j = 0; j < grid->connectivityMatrix->columns; j++) {
            linalgcu_matrix_set_element(grid->connectivityMatrix, -1.0f, i, j);
        }
    }
    linalgcu_matrix_copy_to_device(grid->connectivityMatrix, LINALGCU_FALSE, stream);

    // fill intermediate connectivity and elementalResidual matrices
    linalgcu_matrix_data_t id[3], x[3], y[3];
    linalgcu_matrix_data_t temp;
    fastect_basis_t basis[3];

    for (linalgcu_size_t k = 0; k < grid->mesh->elementCount; k++) {
        // get vertices for element
        for (linalgcu_size_t i = 0; i < 3; i++) {
            linalgcu_matrix_get_element(grid->mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(grid->mesh->vertices, &x[i],
                (linalgcu_size_t)id[i], 0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &y[i],
                (linalgcu_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        fastect_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        fastect_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        fastect_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // set connectivity and elemental residual matrix elements
        for (linalgcu_size_t i = 0; i < 3; i++) {
            for (linalgcu_size_t j = 0; j < 3; j++) {
                // get current element count
                linalgcu_matrix_get_element(elementCount, &temp,
                    (linalgcu_size_t)id[i], (linalgcu_size_t)id[j]);

                // set connectivity element
                linalgcu_matrix_set_element(connectivityMatrix,
                    (linalgcu_matrix_data_t)k, (linalgcu_size_t)id[i],
                    (linalgcu_size_t)(id[j] + connectivityMatrix->rows * temp));

                // set elemental residual element
                linalgcu_matrix_set_element(elementalResidualMatrix,
                    calc_residual_integral(x[0], y[0], x[1], y[1], x[2], y[2],
                        basis[i]->coefficients[0], basis[i]->coefficients[1],
                        basis[i]->coefficients[2], basis[j]->coefficients[0],
                        basis[j]->coefficients[1], basis[j]->coefficients[2]),
                    (linalgcu_size_t)id[i],
                    (linalgcu_size_t)(id[j] + connectivityMatrix->rows * temp));

                // increment element count
                elementCount->hostData[(linalgcu_size_t)id[i] + (linalgcu_size_t)id[j] *
                    elementCount->rows] += 1.0f;
            }
        }

        // cleanup
        fastect_basis_release(&basis[0]);
        fastect_basis_release(&basis[1]);
        fastect_basis_release(&basis[2]);
    }

    // upload intermediate matrices
    linalgcu_matrix_copy_to_device(connectivityMatrix, LINALGCU_TRUE, stream);
    linalgcu_matrix_copy_to_device(elementalResidualMatrix, LINALGCU_TRUE, stream);

    // reduce matrices
    dim3 blocks(connectivityMatrix->rows / LINALGCU_BLOCK_SIZE, 1);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    reduce_residual_matrices<<<blocks, threads, 0, stream>>>(
        grid->connectivityMatrix->deviceData,
        grid->elementalResidualMatrix->deviceData,
        connectivityMatrix->deviceData,
        elementalResidualMatrix->deviceData,
        grid->systemMatrix2D->columnIds,
        grid->connectivityMatrix->rows, grid->connectivityMatrix->rows);

    // update residual matrix
    error = fastect_grid_update_residual_matrix(grid, gamma, stream);

    // cleanup
    linalgcu_matrix_release(&elementCount);
    linalgcu_matrix_release(&connectivityMatrix);
    linalgcu_matrix_release(&elementalResidualMatrix);

    return LINALGCU_SUCCESS;
}

// update_system_matrix_kernel
__global__ void update_system_matrix_kernel(linalgcu_matrix_data_t* systemMatrixValues,
    linalgcu_column_id_t* systemMatrixColumnIds,
    linalgcu_matrix_data_t* gradientMatrixTransposedValues,
    linalgcu_column_id_t* gradientMatrixTransposedColumnIds,
    linalgcu_matrix_data_t* gradientMatrixTransposed,
    linalgcu_matrix_data_t* gamma,
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
            50E-3f * exp10f(gamma[id / 2] / 10.0f) * area[id / 2] *
            gradientMatrixTransposed[j + id * gradientMatrixTransposedRows] :
            0.0f;
    }

    // set element
    systemMatrixValues[i * LINALGCU_BLOCK_SIZE + (blockIdx.y * blockDim.y + threadIdx.y)] =
        element;
}

// update system matrix
LINALGCU_EXTERN_C
linalgcu_error_t fastect_grid_update_2D_system_matrix(fastect_grid_t grid,
    linalgcu_matrix_t gamma, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(grid->systemMatrix2D->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_system_matrix_kernel<<<blocks, threads, 0, stream>>>(
        grid->systemMatrix2D->values,
        grid->systemMatrix2D->columnIds,
        grid->gradientMatrixTransposedSparse->values,
        grid->gradientMatrixTransposedSparse->columnIds,
        grid->gradientMatrixTransposed->deviceData,
        gamma->deviceData,
        grid->area->deviceData,
        grid->gradientMatrixTransposed->rows);

    return LINALGCU_SUCCESS;
}

// update residual matrix kernel
__global__ void update_residual_matrix_kernel(linalgcu_matrix_data_t* residualMatrixValues,
    linalgcu_column_id_t* residualMatrixColumnIds,
    linalgcu_column_id_t* systemMatrixColumnIds,
    linalgcu_matrix_data_t* connectivityMatrix,
    linalgcu_matrix_data_t* elementalResidualMatrix,
    linalgcu_matrix_data_t* gamma, linalgcu_size_t rows) {
    // get ids
    linalgcu_size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcu_size_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get columnId
    linalgcu_column_id_t columnId = systemMatrixColumnIds[row * LINALGCU_BLOCK_SIZE + column];

    // set column id
    residualMatrixColumnIds[row * LINALGCU_BLOCK_SIZE + column] = columnId;

    // check column id
    if (columnId == -1) {
        return;
    }

    // calc residual matrix element
    linalgcu_matrix_data_t value = 0.0f;
    linalgcu_column_id_t elementId = -1;
    for (int k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        // get element id
        elementId = (linalgcu_column_id_t)connectivityMatrix[row +
            (column + k * LINALGCU_BLOCK_SIZE) * rows];

        value += elementId != -1 ? elementalResidualMatrix[row +
            (column + k * LINALGCU_BLOCK_SIZE) * rows] *
            50E-3f * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    residualMatrixValues[row * LINALGCU_BLOCK_SIZE + column] = value;
}

// update residual matrix
LINALGCU_EXTERN_C
linalgcu_error_t fastect_grid_update_residual_matrix(fastect_grid_t grid,
    linalgcu_matrix_t gamma, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (gamma == NULL)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(grid->residualMatrix->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_residual_matrix_kernel<<<blocks, threads, 0, stream>>>(
        grid->residualMatrix->values, grid->residualMatrix->columnIds,
        grid->systemMatrix2D->columnIds, grid->connectivityMatrix->deviceData,
        grid->elementalResidualMatrix->deviceData, gamma->deviceData,
        grid->connectivityMatrix->rows);

    return LINALGCU_SUCCESS;
}
