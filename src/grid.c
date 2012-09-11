// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create solver grid
linalgcu_error_t fastect_grid_create(fastect_grid_t* gridPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_matrix_t gamma,
    linalgcu_size_t numHarmonics, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((gridPointer == NULL) || (mesh == NULL) || (electrodes == NULL) || (gamma == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init grid pointer
    *gridPointer = NULL;

    // create grid struct
    fastect_grid_t grid = malloc(sizeof(fastect_grid_s));

    // check success
    if (grid == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    grid->mesh = mesh;
    grid->electrodes = electrodes;
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
    grid->systemMatrices = malloc(sizeof(linalgcu_sparse_matrix_t) * (grid->numHarmonics + 1));

    // check success
    if (grid->systemMatrices == NULL) {
        // cleanup
        fastect_grid_release(&grid);

        return LINALGCU_ERROR;
    }

    // create system matrices
    error = LINALGCU_SUCCESS;
    for (linalgcu_size_t i = 0; i < grid->numHarmonics + 1; i++) {
        error |= linalgcu_sparse_matrix_create_empty(&grid->systemMatrices[i],
            grid->mesh->vertexCount, grid->mesh->vertexCount, stream);
    }

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
    error |= fastect_grid_init_exitation_matrix(grid, 1.0, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_grid_release(&grid);

        return error;
    }

    // update system matrices
    error = fastect_grid_update_system_matrices(grid, gamma, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_grid_release(&grid);

        return error;
    }

    // set grid pointer
    *gridPointer = grid;

    return LINALGCU_SUCCESS;
}

// release solver grid
linalgcu_error_t fastect_grid_release(fastect_grid_t* gridPointer) {
    // check input
    if ((gridPointer == NULL) || (*gridPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get grid
    fastect_grid_t grid = *gridPointer;

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
        for (linalgcu_size_t i = 0; i < grid->numHarmonics + 1; i++) {
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
linalgcu_error_t fastect_grid_init_2D_system_matrix(fastect_grid_t grid, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // calc initial system matrix
    // create matrices
    linalgcu_matrix_t systemMatrix, gradientMatrix, sigmaMatrix;
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
    linalgcu_matrix_data_t x[3], y[3];
    linalgcu_matrix_data_t id[3];
    fastect_basis_t basis[3];
    linalgcu_matrix_data_t area;

    for (linalgcu_size_t k = 0; k < grid->mesh->elementCount; k++) {
        // get vertices for element
        for (linalgcu_size_t i = 0; i < 3; i++) {
            linalgcu_matrix_get_element(grid->mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(grid->mesh->vertices, &x[i], (linalgcu_size_t)id[i], 0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &y[i], (linalgcu_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        fastect_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        fastect_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        fastect_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        for (linalgcu_size_t i = 0; i < 3; i++) {
            linalgcu_matrix_set_element(gradientMatrix,
                basis[i]->gradient[0], 2 * k, (linalgcu_size_t)id[i]);
            linalgcu_matrix_set_element(gradientMatrix,
                basis[i]->gradient[1], 2 * k + 1, (linalgcu_size_t)id[i]);
            linalgcu_matrix_set_element(grid->gradientMatrixTransposed,
                basis[i]->gradient[0], (linalgcu_size_t)id[i], 2 * k);
            linalgcu_matrix_set_element(grid->gradientMatrixTransposed,
                basis[i]->gradient[1], (linalgcu_size_t)id[i], 2 * k + 1);
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
    error  = linalgcu_matrix_copy_to_device(gradientMatrix, LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(grid->gradientMatrixTransposed,
        LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(sigmaMatrix, LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(grid->area, LINALGCU_TRUE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&systemMatrix);
        linalgcu_matrix_release(&gradientMatrix);
        linalgcu_matrix_release(&sigmaMatrix);

        return LINALGCU_ERROR;
    }

    // calc system matrix
    linalgcu_matrix_t temp = NULL;
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
    for (linalgcu_size_t i = 0; i < grid->numHarmonics + 1; i++) {
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
linalgcu_error_t fastect_grid_update_system_matrices(fastect_grid_t grid,
    linalgcu_matrix_t gamma, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;
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
    linalgcu_matrix_data_t alpha = 0.0f;
    for (linalgcu_size_t n = 1; n < grid->numHarmonics + 1; n++) {
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

linalgcu_matrix_data_t fastect_grid_angle(linalgcu_matrix_data_t x, linalgcu_matrix_data_t y) {
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

linalgcu_matrix_data_t fastect_grid_integrate_basis(linalgcu_matrix_data_t* node,
    linalgcu_matrix_data_t* left, linalgcu_matrix_data_t* right,
    linalgcu_matrix_data_t* start, linalgcu_matrix_data_t* end) {
    // integral
    linalgcu_matrix_data_t integral = 0.0f;

    // calc angle of node
    linalgcu_matrix_data_t angle = fastect_grid_angle(node[0], node[1]);
    linalgcu_matrix_data_t radius = sqrt(start[0] * start[0] + start[1] * start[1]);

    // calc s parameter
    linalgcu_matrix_data_t sleft = radius * (fastect_grid_angle(left[0], left[1]) - angle);
    linalgcu_matrix_data_t sright = radius * (fastect_grid_angle(right[0], right[1]) - angle);
    linalgcu_matrix_data_t sstart = radius * (fastect_grid_angle(start[0], start[1]) - angle);
    linalgcu_matrix_data_t send = radius * (fastect_grid_angle(end[0], end[1]) - angle);

    // correct s parameter
    sleft -= (sleft > sright) && (sleft > 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sright += (sleft > sright) && (sleft < 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sstart -= (sstart > send) && (sstart > 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sstart += (sstart > send) && (sstart < 0.0f) ? radius * 2.0f * M_PI : 0.0f;

    // integrate left triangle
    if ((sstart < 0.0f) && (send > sleft)) {
        if ((send >= 0.0f) && (sstart <= sleft)) {
            integral = -0.5f * sleft;
        }
        else if ((send >= 0.0f) && (sstart > sleft)) {
            integral = -(sstart - 0.5 * sstart * sstart / sleft);
        }
        else if ((send < 0.0f) && (sstart <= sleft)) {
            integral = (send - 0.5 * send * send / sleft) -
                       (sleft - 0.5 * sleft * sleft / sleft);
        }
        else if ((send < 0.0f) && (sstart > sleft)) {
            integral = (send - 0.5 * send * send / sleft) -
                       (sstart - 0.5 * sstart * sstart / sleft);
        }
    }

    // integrate right triangle
    if ((send > 0.0f) && (sright > sstart)) {
        if ((sstart <= 0.0f) && (send >= sright)) {
            integral += 0.5f * sright;
        }
        else if ((sstart <= 0.0f) && (send < sright)) {
            integral += (send - 0.5f * send * send / sright);
        }
        else if ((sstart > 0.0f) && (send >= sright)) {
            integral += (sright - 0.5f * sright * sright / sright) -
                        (sstart - 0.5f * sstart * sstart / sright);
        }
        else if ((sstart > 0.0f) && (send < sright)) {
            integral += (send - 0.5f * send * send / sright) -
                        (sstart - 0.5f * sstart * sstart / sright);
        }
    }

    return integral;
}

// init exitation matrix
linalgcu_error_t fastect_grid_init_exitation_matrix(fastect_grid_t grid,
    linalgcu_matrix_data_t current, cudaStream_t stream) {
    // check input
    if (grid == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // fill exitation_matrix matrix
    linalgcu_matrix_data_t id[3];
    linalgcu_matrix_data_t node[2], left[2], right[2];

    for (int i = 0; i < grid->mesh->boundaryCount; i++) {
        for (int j = 0; j < grid->electrodes->count; j++) {
            // get boundary node id
            linalgcu_matrix_get_element(grid->mesh->boundary, &id[0],
                i - 1 < 0 ? grid->mesh->boundaryCount - 1 : i - 1, 0);
            linalgcu_matrix_get_element(grid->mesh->boundary, &id[1], i, 0);
            linalgcu_matrix_get_element(grid->mesh->boundary, &id[2],
                (i + 1) % grid->mesh->boundaryCount, 0);

            // get coordinates
            linalgcu_matrix_get_element(grid->mesh->vertices, &left[0], (linalgcu_size_t)id[0],
                0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &left[1], (linalgcu_size_t)id[0],
                1);
            linalgcu_matrix_get_element(grid->mesh->vertices, &node[0], (linalgcu_size_t)id[1],
                0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &node[1], (linalgcu_size_t)id[1],
                1);
            linalgcu_matrix_get_element(grid->mesh->vertices, &right[0], (linalgcu_size_t)id[2],
                0);
            linalgcu_matrix_get_element(grid->mesh->vertices, &right[1], (linalgcu_size_t)id[2],
                1);

            // calc element
            linalgcu_matrix_set_element(grid->excitationMatrix,
                fastect_grid_integrate_basis(node, left, right,
                    &grid->electrodes->electrodesStart[j * 2],
                    &grid->electrodes->electrodesEnd[j * 2]) *
                    current / grid->electrodes->width, (linalgcu_size_t)id[1], j);
        }
    }

    // upload matrix
    linalgcu_matrix_copy_to_device(grid->excitationMatrix, LINALGCU_TRUE, stream);

    return LINALGCU_SUCCESS;
}
