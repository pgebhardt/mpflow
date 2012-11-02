// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fasteit.h"

// create forward_solver
linalgcuError_t fasteit_forward_solver_create(fasteitForwardSolver_t* solverPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrix_t measurmentPattern,
    linalgcuMatrix_t drivePattern, linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrixData_t numHarmonics, linalgcuMatrixData_t sigmaRef, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drivePattern == NULL) || (measurmentPattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fasteitForwardSolver_t self = malloc(sizeof(fasteitForwardSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->model = NULL;
    self->conjugateSolver = NULL;
    self->driveCount = driveCount;
    self->measurmentCount = measurmentCount;
    self->jacobian = NULL;
    self->voltage = NULL;
    self->phi = NULL;
    self->excitation = NULL;
    self->voltageCalculation = NULL;
    self->area = NULL;
    self->gradientMatrixSparse = NULL;

    // create model
    error = fasteit_model_create(&self->model, mesh, electrodes, sigmaRef, numHarmonics,
        handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return error;
    }

    // create conjugate solver
    error  = fasteit_sparse_conjugate_solver_create(&self->conjugateSolver,
        mesh->vertexCount, self->driveCount + self->measurmentCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return error;
    }

    // create matrices
    error |= linalgcu_matrix_create(&self->jacobian,
        measurmentPattern->columns * drivePattern->columns, mesh->elementCount, stream);
    error |= linalgcu_matrix_create(&self->voltage, measurmentCount, driveCount, stream);
    error |= linalgcu_matrix_create(&self->voltageCalculation,
        self->measurmentCount, mesh->vertexCount, stream);
    error |= linalgcu_matrix_create(&self->area, mesh->elementCount, 1, stream);

    // create matrix buffer
    self->phi = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    self->excitation = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));

    // check success
    if ((self->phi == NULL) || (self->excitation == NULL)) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // create matrices
    for (linalgcuSize_t i = 0; i < numHarmonics + 1; i++) {
        error |= linalgcu_matrix_create(&self->phi[i], mesh->vertexCount,
            self->driveCount + self->measurmentCount, stream);
        error |= linalgcu_matrix_create(&self->excitation[i], mesh->vertexCount,
            self->driveCount + self->measurmentCount, stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return error;
    }

    // create pattern matrix
    linalgcuMatrix_t pattern = NULL;
    error |= linalgcu_matrix_create(&pattern, drivePattern->rows,
       self->driveCount + self->measurmentCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return error;
    }

    // fill pattern matrix with drive pattern
    linalgcuMatrixData_t value = 0.0f;
    for (linalgcuSize_t i = 0; i < pattern->rows; i++) {
        for (linalgcuSize_t j = 0; j < self->driveCount; j++) {
            // get value
            linalgcu_matrix_get_element(drivePattern, &value, i, j);

            // set value
            linalgcu_matrix_set_element(pattern, value, i, j);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (linalgcuSize_t i = 0; i < pattern->rows; i++) {
        for (linalgcuSize_t j = 0; j < self->measurmentCount; j++) {
            // get value
            linalgcu_matrix_get_element(measurmentPattern, &value, i, j);

            // set value
            linalgcu_matrix_set_element(pattern, -value, i, j + self->driveCount);
        }
    }

    linalgcu_matrix_copy_to_device(pattern, stream);

    // calc excitation matrices
    error = fasteit_model_calc_excitaions(self->model, self->excitation, pattern,
        handle, stream);

    // cleanup
    linalgcu_matrix_release(&pattern);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return error;
    }

    // calc voltage calculation matrix
    linalgcuMatrixData_t alpha = -1.0f, beta = 0.0f;

    // one prerun for cublas
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        self->model->excitationMatrix->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        self->model->excitationMatrix->deviceData, self->model->excitationMatrix->rows,
        &beta, self->voltageCalculation->deviceData, self->voltageCalculation->rows);

    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        self->model->excitationMatrix->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        self->model->excitationMatrix->deviceData, self->model->excitationMatrix->rows,
        &beta, self->voltageCalculation->deviceData, self->voltageCalculation->rows)
        != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // create matrices
    linalgcuMatrix_t gradientMatrix;
    error = linalgcu_matrix_create(&gradientMatrix,
        2 * mesh->elementCount, mesh->vertexCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // calc gradient matrix
    linalgcuMatrixData_t x[3], y[3];
    linalgcuMatrixData_t id[3];
    fasteitBasis_t basis[3];
    linalgcuMatrixData_t area;

    for (linalgcuSize_t k = 0; k < mesh->elementCount; k++) {
        // get vertices for element
        for (linalgcuSize_t i = 0; i < 3; i++) {
            linalgcu_matrix_get_element(mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(mesh->vertices, &x[i], (linalgcuSize_t)id[i], 0);
            linalgcu_matrix_get_element(mesh->vertices, &y[i], (linalgcuSize_t)id[i], 1);
        }

        // calc corresponding basis functions
        fasteit_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        fasteit_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        fasteit_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        for (linalgcuSize_t i = 0; i < 3; i++) {
            linalgcu_matrix_set_element(gradientMatrix,
                basis[i]->coefficients[1], 2 * k, (linalgcuSize_t)id[i]);
            linalgcu_matrix_set_element(gradientMatrix,
                basis[i]->coefficients[2], 2 * k + 1, (linalgcuSize_t)id[i]);
        }

        // calc area of element
        area = 0.5 * fabs((x[1] - x[0]) * (y[2] - y[0]) -
            (x[2] - x[0]) * (y[1] - y[0]));

        linalgcu_matrix_set_element(self->area, area, k, 0);

        // cleanup
        fasteit_basis_release(&basis[0]);
        fasteit_basis_release(&basis[1]);
        fasteit_basis_release(&basis[2]);
    }

    // copy matrices to device
    error  = linalgcu_matrix_copy_to_device(gradientMatrix, stream);
    error |= linalgcu_matrix_copy_to_device(self->area, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&gradientMatrix);
        fasteit_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // create sparse matrices
    error = linalgcu_sparse_matrix_create(&self->gradientMatrixSparse, gradientMatrix, stream);

    // cleanup
    linalgcu_matrix_release(&gradientMatrix);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fasteit_forward_solver_release(fasteitForwardSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fasteitForwardSolver_t self = *solverPointer;

    // cleanup
    linalgcu_matrix_release(&self->jacobian);
    linalgcu_matrix_release(&self->voltage);
    linalgcu_matrix_release(&self->voltageCalculation);
    linalgcu_matrix_release(&self->area);
    linalgcu_sparse_matrix_release(&self->gradientMatrixSparse);

    if (self->phi != NULL) {
        for (linalgcuSize_t i = 0; i < self->model->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&self->phi[i]);
        }
        free(self->phi);
    }
    if (self->excitation != NULL) {
        for (linalgcuSize_t i = 0; i < self->model->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&self->excitation[i]);
        }
        free(self->excitation);
    }
    fasteit_model_release(&self->model);
    fasteit_sparse_conjugate_solver_release(&self->conjugateSolver);

    // free struct
    free(self);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcuError_t fasteit_forward_solver_solve(fasteitForwardSolver_t self,
    linalgcuMatrix_t gamma, linalgcuSize_t steps, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // update system matrix
    error  = fasteit_model_update(self->model, gamma, handle, stream);

    // solve for ground mode
    // solve for drive phi
    error |= fasteit_sparse_conjugate_solver_solve(self->conjugateSolver,
        self->model->systemMatrices[0], self->phi[0], self->excitation[0],
        steps, LINALGCU_TRUE, stream);

    // solve for higher harmonics
    for (linalgcuSize_t n = 1; n < self->model->numHarmonics + 1; n++) {
        // solve for drive phi
        error |= fasteit_sparse_conjugate_solver_solve(self->conjugateSolver,
            self->model->systemMatrices[n], self->phi[n], self->excitation[n],
            steps, LINALGCU_FALSE, stream);
    }

    // calc jacobian
    error |= fasteit_forward_solver_calc_jacobian(self, gamma, 0, LINALGCU_FALSE, stream);
    for (linalgcuSize_t n = 1; n < self->model->numHarmonics + 1; n++) {
        error |= fasteit_forward_solver_calc_jacobian(self, gamma, n, LINALGCU_TRUE, stream);
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, self->voltageCalculation->rows,
        self->driveCount, self->voltageCalculation->columns, &alpha,
        self->voltageCalculation->deviceData, self->voltageCalculation->rows,
        self->phi[0]->deviceData, self->phi[0]->rows, &beta,
        self->voltage->deviceData, self->voltage->rows);

    // add harmonic voltages
    beta = 1.0f;
    for (linalgcuSize_t n = 1; n < self->model->numHarmonics + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, self->voltageCalculation->rows,
            self->driveCount, self->voltageCalculation->columns, &alpha,
            self->voltageCalculation->deviceData, self->voltageCalculation->rows,
            self->phi[n]->deviceData, self->phi[n]->rows, &beta,
            self->voltage->deviceData, self->voltage->rows);
    }

    return error;
}
