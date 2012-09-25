// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create solver grid
linalgcuError_t fastect_grid_create(fastectGrid_t* gridPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((gridPointer == NULL) || (mesh == NULL) || (electrodes == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init grid pointer
    *gridPointer = NULL;

    // create grid struct
    fastectGrid_t grid = malloc(sizeof(fastectGrid_s));

    // check success
    if (grid == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    grid->mesh = mesh;
    grid->electrodes = electrodes;
    grid->sigmaRef = sigmaRef;
    grid->systemMatrices = NULL;
    grid->systemMatrix2D = NULL;
    grid->residualMatrix = NULL;
    grid->excitationMatrix = NULL;
    grid->gradientMatrixTransposed = NULL;
    grid->gradientMatrixTransposedSparse = NULL;
    grid->connectivityMatrix = NULL;
    grid->elementalResidualMatrix = NULL;
    grid->area = NULL;
    grid->numHarmonics = numHarmonics;

    // create matrices
    error  = linalgcu_matrix_create(&grid->excitationMatrix,
        grid->mesh->vertexCount, grid->electrodes->count, stream);
    error |= linalgcu_matrix_create(&grid->gradientMatrixTransposed,
        grid->mesh->vertexCount, 2 * grid->mesh->elementCount, stream);
    error |= linalgcu_matrix_create(&grid->connectivityMatrix, grid->mesh->vertexCount,
        LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&grid->elementalResidualMatrix, grid->mesh->vertexCount,
        LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&grid->area, grid->mesh->elementCount, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // create system matrices
    grid->systemMatrices = malloc(sizeof(linalgcuSparseMatrix_t) * (grid->numHarmonics + 1));

    // check success
    if (grid->systemMatrices == NULL) {
        // cleanup
        fastect_grid_release(&grid);

        return LINALGCU_ERROR;
    }

    // create system matrices
    error = LINALGCU_SUCCESS;
    for (linalgcuSize_t i = 0; i < grid->numHarmonics + 1; i++) {
        error |= linalgcu_sparse_matrix_create_empty(&grid->systemMatrices[i],
            grid->mesh->vertexCount, grid->mesh->vertexCount, stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_grid_release(&grid);

        return error;
    }

    // create initial gamma
    linalgcuMatrix_t gamma = NULL;
    error = linalgcu_matrix_create(&gamma, grid->mesh->elementCount, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_grid_release(&grid);

        return error;
    }

    // init system matrix
    error  = fastect_grid_init_2D_system_matrix(grid, handle, stream);

    // init residual matrix
    error |= fastect_grid_init_residual_matrix(grid, gamma, stream);

    // init excitaion matrix
    error |= fastect_grid_init_exitation_matrix(grid, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&gamma);
        fastect_grid_release(&grid);

        return error;
    }

    // update system matrices
    error = fastect_grid_update_system_matrices(grid, gamma, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&gamma);
        fastect_grid_release(&grid);

        return error;
    }

    // set grid pointer
    *gridPointer = grid;

    return LINALGCU_SUCCESS;
}

// release solver grid
linalgcuError_t fastect_grid_release(fastectGrid_t* gridPointer) {
    // check input
    if ((gridPointer == NULL) || (*gridPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get grid
    fastectGrid_t grid = *gridPointer;

    // cleanup
    fastect_mesh_release(&grid->mesh);
    fastect_electrodes_release(&grid->electrodes);
    linalgcu_sparse_matrix_release(&grid->systemMatrix2D);
    linalgcu_sparse_matrix_release(&grid->residualMatrix);
    linalgcu_matrix_release(&grid->excitationMatrix);
    linalgcu_matrix_release(&grid->gradientMatrixTransposed);
    linalgcu_sparse_matrix_release(&grid->gradientMatrixTransposedSparse);
    linalgcu_sparse_matrix_release(&grid->gradientMatrixSparse);
    linalgcu_matrix_release(&grid->connectivityMatrix);
    linalgcu_matrix_release(&grid->elementalResidualMatrix);
    linalgcu_matrix_release(&grid->area);

    if (grid->systemMatrices != NULL) {
        for (linalgcuSize_t i = 0; i < grid->numHarmonics + 1; i++) {
            linalgcu_sparse_matrix_release(&grid->systemMatrices[i]);
        }
        free(grid->systemMatrices);
    }

    // free struct
    free(grid);

    // set grid pointer to NULL
    *gridPointer = NULL;

    return LINALGCU_SUCCESS;
}

// init system matrix 2D
linalgcuError_t fastect_grid_init_2D_system_matrix(fastectGrid_t grid, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calc initial system matrix
    // create matrices
    linalgcuMatrix_t systemMatrix, gradientMatrix, sigmaMatrix;
    error = linalgcu_matrix_create(&systemMatrix,
        grid->mesh->vertexCount, grid->mesh->vertexCount, stream);
    error += linalgcu_matrix_create(&gradientMatrix,
        2 * grid->mesh->elementCount, grid->mesh->vertexCount, stream);
    error += linalgcu_matrix_unity(&sigmaMatrix,
        2 * grid->mesh->elementCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc gradient matrix
    linalgcuMatrixData_t x[3], y[3];
    linalgcuMatrixData_t id[3];
    fastectBasis_t basis[3];
    linalgcuMatrixData_t area;

    for (linalgcuSize_t k = 0; k < grid->mesh->elementCount; k++) {
        // get vertices for element
        for (linalgcuSize_t i = 0; i < 3; i++) {
            linalgcu_matrix_get_element(grid->mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(grid->mesh->vertices, &x[i], (linalgcuSize_t)id[i], 0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &y[i], (linalgcuSize_t)id[i], 1);
        }

        // calc corresponding basis functions
        fastect_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        fastect_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        fastect_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        for (linalgcuSize_t i = 0; i < 3; i++) {
            linalgcu_matrix_set_element(gradientMatrix,
                basis[i]->gradient[0], 2 * k, (linalgcuSize_t)id[i]);
            linalgcu_matrix_set_element(gradientMatrix,
                basis[i]->gradient[1], 2 * k + 1, (linalgcuSize_t)id[i]);
            linalgcu_matrix_set_element(grid->gradientMatrixTransposed,
                basis[i]->gradient[0], (linalgcuSize_t)id[i], 2 * k);
            linalgcu_matrix_set_element(grid->gradientMatrixTransposed,
                basis[i]->gradient[1], (linalgcuSize_t)id[i], 2 * k + 1);
        }

        // calc area of element
        area = 0.5 * fabs((x[1] - x[0]) * (y[2] - y[0]) -
            (x[2] - x[0]) * (y[1] - y[0]));

        linalgcu_matrix_set_element(grid->area, area, k, 0);
        linalgcu_matrix_set_element(sigmaMatrix, area, 2 * k, 2 * k);
        linalgcu_matrix_set_element(sigmaMatrix, area, 2 * k + 1, 2 * k + 1);

        // cleanup
        fastect_basis_release(&basis[0]);
        fastect_basis_release(&basis[1]);
        fastect_basis_release(&basis[2]);
    }

    // copy matrices to device
    error  = linalgcu_matrix_copy_to_device(gradientMatrix, stream);
    error |= linalgcu_matrix_copy_to_device(grid->gradientMatrixTransposed, stream);
    error |= linalgcu_matrix_copy_to_device(sigmaMatrix, stream);
    error |= linalgcu_matrix_copy_to_device(grid->area, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&systemMatrix);
        linalgcu_matrix_release(&gradientMatrix);
        linalgcu_matrix_release(&sigmaMatrix);

        return LINALGCU_ERROR;
    }

    // calc system matrix
    linalgcuMatrix_t temp = NULL;
    error = linalgcu_matrix_create(&temp, grid->mesh->vertexCount,
        2 * grid->mesh->elementCount, stream);

    // one prerun cublas to get ready
    linalgcu_matrix_multiply(temp, grid->gradientMatrixTransposed,
        sigmaMatrix, handle, stream);

    error |= linalgcu_matrix_multiply(temp, grid->gradientMatrixTransposed,
        sigmaMatrix, handle, stream);
    error |= linalgcu_matrix_multiply(systemMatrix, temp, gradientMatrix, handle, stream);
    cudaStreamSynchronize(stream);
    linalgcu_matrix_release(&temp);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&systemMatrix);

        return LINALGCU_ERROR;
    }

    // create sparse matrices
    error  = linalgcu_sparse_matrix_create(&grid->systemMatrix2D, systemMatrix, stream);
    error |= linalgcu_sparse_matrix_create(&grid->residualMatrix, systemMatrix, stream);
    error |= linalgcu_sparse_matrix_create(&grid->gradientMatrixSparse,
        gradientMatrix, stream);
    error |= linalgcu_sparse_matrix_create(&grid->gradientMatrixTransposedSparse,
        grid->gradientMatrixTransposed, stream);
    for (linalgcuSize_t i = 0; i < grid->numHarmonics + 1; i++) {
        error |= linalgcu_sparse_matrix_create(&grid->systemMatrices[i], systemMatrix, stream);
    }

    // cleanup
    linalgcu_matrix_release(&sigmaMatrix);
    linalgcu_matrix_release(&gradientMatrix);
    linalgcu_matrix_release(&systemMatrix);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// update system matrix
linalgcuError_t fastect_grid_update_system_matrices(fastectGrid_t grid,
    linalgcuMatrix_t gamma, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;
    cublasStatus_t cublasError = CUBLAS_STATUS_SUCCESS;

    // update 2d systemMatrix
    error  = fastect_grid_update_2D_system_matrix(grid, gamma, stream);

    // update residual matrix
    error |= fastect_grid_update_residual_matrix(grid, gamma, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // set first system matrix to 2d system matrix
    cublasError = cublasScopy(handle, grid->systemMatrix2D->rows * LINALGCU_BLOCK_SIZE,
        grid->systemMatrix2D->values, 1, grid->systemMatrices[0]->values, 1);

    // create harmonic system matrices
    linalgcuMatrixData_t alpha = 0.0f;
    for (linalgcuSize_t n = 1; n < grid->numHarmonics + 1; n++) {
        // calc alpha
        alpha = (2.0f * n * M_PI / grid->mesh->height) *
            (2.0f * n * M_PI / grid->mesh->height);

        // init system matrix with 2d system matrix
        cublasError |= cublasScopy(handle, grid->systemMatrix2D->rows * LINALGCU_BLOCK_SIZE,
            grid->systemMatrix2D->values, 1, grid->systemMatrices[n]->values, 1);

        // add alpha * residualMatrix
        cublasError |= cublasSaxpy(handle, grid->systemMatrix2D->rows * LINALGCU_BLOCK_SIZE, &alpha,
            grid->residualMatrix->values, 1, grid->systemMatrices[n]->values, 1);
    }

    // check error
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

linalgcuMatrixData_t fastect_grid_angle(linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    if (x > 0.0f) {
        return atan(y / x);
    }
    else if ((x < 0.0f) && (y >= 0.0f)) {
        return atan(y / x) + M_PI;
    }
    else if ((x < 0.0f) && (y < 0.0f)) {
        return atan(y / x) - M_PI;
    }
    else if ((x == 0.0f) && (y > 0.0f)) {
        return M_PI / 2.0f;
    }
    else if ((x == 0.0f) && (y < 0.0f)) {
        return - M_PI / 2.0f;
    }
    else {
        return 0.0f;
    }
}

linalgcuMatrixData_t fastect_grid_integrate_basis(linalgcuMatrixData_t* start,
    linalgcuMatrixData_t* end, linalgcuMatrixData_t* electrodeStart,
    linalgcuMatrixData_t* electrodeEnd) {
    // integral
    linalgcuMatrixData_t integral = 0.0f;

    // calc radius
    linalgcuMatrixData_t radius = sqrt(electrodeStart[0] * electrodeStart[0]
        + electrodeStart[1] * electrodeStart[1]);

    // calc angle
    linalgcuMatrixData_t angleStart = fastect_grid_angle(start[0], start[1]);
    linalgcuMatrixData_t angleEnd = fastect_grid_angle(end[0], end[1]) - angleStart;
    linalgcuMatrixData_t angleElectrodeStart = fastect_grid_angle(electrodeStart[0], electrodeStart[1]) - angleStart;
    linalgcuMatrixData_t angleElectrodeEnd = fastect_grid_angle(electrodeEnd[0], electrodeEnd[1]) - angleStart;

    // correct angle
    angleEnd += (angleEnd < M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeStart += (angleElectrodeStart < M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeEnd += (angleElectrodeEnd < M_PI) ? 2.0f * M_PI : 0.0f;
    angleEnd -= (angleEnd > M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeStart -= (angleElectrodeStart > M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeEnd -= (angleElectrodeEnd > M_PI) ? 2.0f * M_PI : 0.0f;

    // calc parameter
    linalgcuMatrixData_t sEnd = radius * angleEnd;
    linalgcuMatrixData_t sElectrodeStart = radius * angleElectrodeStart;
    linalgcuMatrixData_t sElectrodeEnd = radius * angleElectrodeEnd;

    // integrate left triangle
    if (sEnd < 0.0f) {
        if ((sElectrodeStart < 0.0f) && (sElectrodeEnd > sEnd)) {
            if ((sElectrodeEnd >= 0.0f) && (sElectrodeStart <= sEnd)) {
                integral = -0.5f * sEnd;
            }
            else if ((sElectrodeEnd >= 0.0f) && (sElectrodeStart > sEnd)) {
                integral = -(sElectrodeStart - 0.5 * sElectrodeStart * sElectrodeStart / sEnd);
            }
            else if ((sElectrodeEnd < 0.0f) && (sElectrodeStart <= sEnd)) {
                integral = (sElectrodeEnd - 0.5 * sElectrodeEnd * sElectrodeEnd / sEnd) -
                           (sEnd - 0.5 * sEnd * sEnd / sEnd);
            }
            else if ((sElectrodeEnd < 0.0f) && (sElectrodeStart > sEnd)) {
                integral = (sElectrodeEnd - 0.5 * sElectrodeEnd * sElectrodeEnd / sEnd) -
                           (sElectrodeStart - 0.5 * sElectrodeStart * sElectrodeStart / sEnd);
            }
        }
    }
    else {
        // integrate right triangle
        if ((sElectrodeEnd > 0.0f) && (sEnd > sElectrodeStart)) {
            if ((sElectrodeStart <= 0.0f) && (sElectrodeEnd >= sEnd)) {
                integral = 0.5f * sEnd;
            }
            else if ((sElectrodeStart <= 0.0f) && (sElectrodeEnd < sEnd)) {
                integral = (sElectrodeEnd - 0.5f * sElectrodeEnd * sElectrodeEnd / sEnd);
            }
            else if ((sElectrodeStart > 0.0f) && (sElectrodeEnd >= sEnd)) {
                integral = (sEnd - 0.5f * sEnd * sEnd / sEnd) -
                            (sElectrodeStart - 0.5f * sElectrodeStart * sElectrodeStart / sEnd);
            }
            else if ((sElectrodeStart > 0.0f) && (sElectrodeEnd < sEnd)) {
                integral = (sElectrodeEnd - 0.5f * sElectrodeEnd * sElectrodeEnd / sEnd) -
                            (sElectrodeStart - 0.5f * sElectrodeStart * sElectrodeStart / sEnd);
            }
        }
    }

    return integral;
}

// init exitation matrix
linalgcuError_t fastect_grid_init_exitation_matrix(fastectGrid_t grid,
    cudaStream_t stream) {
    // check input
    if (grid == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // fill exitation_matrix matrix
    linalgcuMatrixData_t id[2];
    linalgcuMatrixData_t node[2], end[2];

    for (int i = 0; i < grid->mesh->boundaryCount; i++) {
        for (int j = 0; j < grid->electrodes->count; j++) {
            // get boundary node id
            linalgcu_matrix_get_element(grid->mesh->boundary, &id[0], i, 0);
            linalgcu_matrix_get_element(grid->mesh->boundary, &id[1], i, 1);

            // get coordinates
            linalgcu_matrix_get_element(grid->mesh->vertices, &node[0], (linalgcuSize_t)id[0],
                0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &node[1], (linalgcuSize_t)id[0],
                1);
            linalgcu_matrix_get_element(grid->mesh->vertices, &end[0], (linalgcuSize_t)id[1],
                0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &end[1], (linalgcuSize_t)id[1],
                1);

            // calc element
            grid->excitationMatrix->hostData[(linalgcuSize_t)id[0] + j * grid->excitationMatrix->rows] +=
                fastect_grid_integrate_basis(node, end,
                    &grid->electrodes->electrodesStart[j * 2],
                    &grid->electrodes->electrodesEnd[j * 2]) / grid->electrodes->width;
            grid->excitationMatrix->hostData[(linalgcuSize_t)id[1] + j * grid->excitationMatrix->rows] +=
                fastect_grid_integrate_basis(end, node,
                    &grid->electrodes->electrodesStart[j * 2],
                    &grid->electrodes->electrodesEnd[j * 2]) / grid->electrodes->width;
        }
    }

    // upload matrix
    linalgcu_matrix_copy_to_device(grid->excitationMatrix, stream);

    return LINALGCU_SUCCESS;
}
