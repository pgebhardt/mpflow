// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create forward_solver
linalgcuError_t fastect_forward_solver_create(fastectForwardSolver_t* solverPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, linalgcuSize_t driveCount, linalgcuSize_t measurmentCount,
    linalgcuMatrix_t drivePattern, linalgcuMatrix_t measurmentPattern, cublasHandle_t handle,
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
    fastectForwardSolver_t self = malloc(sizeof(fastectForwardSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->grid = NULL;
    self->driveSolver = NULL;
    self->measurmentSolver = NULL;
    self->drivePhi = NULL;
    self->measurmentPhi = NULL;
    self->driveF = NULL;
    self->measurmentF = NULL;
    self->voltageCalculation = NULL;

    // create grid
    error = fastect_grid_create(&self->grid, mesh, electrodes, sigmaRef, numHarmonics,
        handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&self);

        return error;
    }

    // create conjugate solver
    error  = fastect_conjugate_sparse_solver_create(&self->driveSolver,
        mesh->vertexCount, driveCount, stream);
    error |= fastect_conjugate_sparse_solver_create(&self->measurmentSolver,
        mesh->vertexCount, measurmentCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&self);

        return error;
    }

    // create matrix buffer
    self->drivePhi = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    self->measurmentPhi = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    self->driveF = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));
    self->measurmentF = malloc(sizeof(linalgcuMatrix_t) * (numHarmonics + 1));

    // check success
    if ((self->drivePhi == NULL) || (self->measurmentPhi == NULL) ||
        (self->driveF == NULL) || (self->measurmentF == NULL)) {
        // cleanup
        fastect_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // create matrices
    for (linalgcuSize_t i = 0; i < numHarmonics + 1; i++) {
        error |= linalgcu_matrix_create(&self->drivePhi[i], mesh->vertexCount,
            driveCount, stream);
        error |= linalgcu_matrix_create(&self->measurmentPhi[i], mesh->vertexCount,
            measurmentCount, stream);
        error |= linalgcu_matrix_create(&self->driveF[i], mesh->vertexCount,
            driveCount, stream);
        error |= linalgcu_matrix_create(&self->measurmentF[i], mesh->vertexCount,
            measurmentCount, stream);
    }

    error |= linalgcu_matrix_create(&self->voltageCalculation,
        measurmentPattern->columns, mesh->vertexCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&self);

        return error;
    }

    // calc excitaion matrices
    for (linalgcuSize_t n = 0; n < numHarmonics + 1; n++) {
        // Run multiply once more to avoid cublas error
        linalgcu_matrix_multiply(self->driveF[n], self->grid->excitationMatrix,
            drivePattern, handle, stream);
        error |= linalgcu_matrix_multiply(self->driveF[n], self->grid->excitationMatrix,
            drivePattern, handle, stream);

        linalgcu_matrix_multiply(self->measurmentF[n], self->grid->excitationMatrix,
            measurmentPattern, handle, stream);
        error |= linalgcu_matrix_multiply(self->measurmentF[n],
            self->grid->excitationMatrix, measurmentPattern, handle, stream);
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    error |= linalgcu_matrix_scalar_multiply(self->driveF[0],
        1.0f / self->grid->mesh->height, stream);
    error |= linalgcu_matrix_scalar_multiply(self->measurmentF[0],
        1.0f / self->grid->mesh->height, stream);

    // calc harmonics
    for (linalgcuSize_t n = 1; n < numHarmonics + 1; n++) {
        error |= linalgcu_matrix_scalar_multiply(self->driveF[n],
            2.0f * sin(n * M_PI * self->grid->electrodes->height /
            self->grid->mesh->height) /
            (n * M_PI * self->grid->electrodes->height), stream);
        error |= linalgcu_matrix_scalar_multiply(self->measurmentF[n],
            2.0f * sin(n * M_PI * self->grid->electrodes->height /
            self->grid->mesh->height) /
            (n * M_PI * self->grid->electrodes->height), stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&self);

        return error;
    }

    // calc voltage calculation matrix
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        self->grid->excitationMatrix->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        self->grid->excitationMatrix->deviceData, self->grid->excitationMatrix->rows,
        &beta, self->voltageCalculation->deviceData, self->voltageCalculation->rows)
        != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&self);

        return LINALGCU_ERROR;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fastect_forward_solver_release(fastectForwardSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectForwardSolver_t self = *solverPointer;

    // cleanup
    if (self->drivePhi != NULL) {
        for (linalgcuSize_t i = 0; i < self->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&self->drivePhi[i]);
        }
        free(self->drivePhi);
    }
    if (self->measurmentPhi != NULL) {
        for (linalgcuSize_t i = 0; i < self->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&self->measurmentPhi[i]);
        }
        free(self->measurmentPhi);
    }
    if (self->driveF != NULL) {
        for (linalgcuSize_t i = 0; i < self->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&self->driveF[i]);
        }
        free(self->driveF);
    }
    if (self->measurmentF != NULL) {
        for (linalgcuSize_t i = 0; i < self->grid->numHarmonics + 1; i++) {
            linalgcu_matrix_release(&self->measurmentF[i]);
        }
        free(self->measurmentF);
    }
    fastect_grid_release(&self->grid);
    fastect_conjugate_sparse_solver_release(&self->driveSolver);
    fastect_conjugate_sparse_solver_release(&self->measurmentSolver);

    // free struct
    free(self);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcuError_t fastect_forward_solver_solve(fastectForwardSolver_t self,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t gamma, linalgcuMatrix_t voltage,
    linalgcuSize_t steps, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (gamma == NULL) || (jacobian == NULL) || (voltage == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // update system matrix
    error  = fastect_grid_update_system_matrices(self->grid, gamma, handle, stream);

    // solve for ground mode
    // solve for drive phi
    error |= fastect_conjugate_sparse_solver_solve(self->driveSolver,
        self->grid->systemMatrices[0], self->drivePhi[0], self->driveF[0],
        steps, LINALGCU_TRUE, stream);

    // solve for measurment phi
    error |= fastect_conjugate_sparse_solver_solve(self->measurmentSolver,
        self->grid->systemMatrices[0], self->measurmentPhi[0], self->measurmentF[0],
        steps, LINALGCU_TRUE, stream);

    // solve for higher harmonics
    for (linalgcuSize_t n = 1; n < self->grid->numHarmonics + 1; n++) {
        // solve for drive phi
        error |= fastect_conjugate_sparse_solver_solve(self->driveSolver,
            self->grid->systemMatrices[n], self->drivePhi[n], self->driveF[n],
            steps, LINALGCU_FALSE, stream);

        // solve for measurment phi
        error |= fastect_conjugate_sparse_solver_solve(self->measurmentSolver,
            self->grid->systemMatrices[n], self->measurmentPhi[n], self->measurmentF[n],
            steps, LINALGCU_FALSE, stream);
    }

    // calc jacobian
    error |= fastect_forward_solver_calc_jacobian(self, jacobian, gamma, 0,
        LINALGCU_FALSE, stream);
    for (linalgcuSize_t n = 1; n < self->grid->numHarmonics + 1; n++) {
        error |= fastect_forward_solver_calc_jacobian(self, jacobian, gamma, n,
            LINALGCU_TRUE, stream);
    }

    // calc voltage
    error |= linalgcu_matrix_multiply(voltage, self->voltageCalculation,
        self->drivePhi[0], handle, stream);

    // set stream
    cublasSetStream(handle, stream);

    // add harmonic voltages
    linalgcuMatrixData_t alpha = 1.0f, beta = 1.0f;
    for (linalgcuSize_t n = 1; n < self->grid->numHarmonics + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, self->voltageCalculation->rows,
            self->drivePhi[n]->columns, self->voltageCalculation->columns, &alpha,
            self->voltageCalculation->deviceData, self->voltageCalculation->rows,
            self->drivePhi[n]->deviceData, self->drivePhi[n]->rows, &beta,
            voltage->deviceData, voltage->rows);
    }

    return error;
}
