// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fasteit.h"

// create solver
linalgcuError_t fasteit_solver_create(fasteitSolver_t* solverPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrix_t measurmentPattern,
    linalgcuMatrix_t drivePattern, linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrixData_t numHarmonics, linalgcuMatrixData_t sigmaRef,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drivePattern == NULL) || (measurmentPattern == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fasteitSolver_t self = malloc(sizeof(fasteitSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->forwardSolver = NULL;
    self->inverseSolver = NULL;
    self->dGamma = NULL;
    self->gamma = NULL;
    self->measuredVoltage = NULL;
    self->calibrationVoltage = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&self->dGamma, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&self->gamma, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&self->measuredVoltage, measurmentCount, driveCount,
        stream);
    error |= linalgcu_matrix_create(&self->calibrationVoltage, measurmentCount, driveCount,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_solver_release(&self);

        return error;
    }

    // create solver
    error  = fasteit_forward_solver_create(&self->forwardSolver, mesh, electrodes,
        measurmentPattern, drivePattern, measurmentCount, driveCount, numHarmonics, sigmaRef,
        handle, stream);
    error |= fasteit_inverse_solver_create(&self->inverseSolver,
        mesh->elementCount, measurmentPattern->columns * drivePattern->columns,
        regularizationFactor, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_solver_release(&self);

        return error;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fasteit_solver_release(fasteitSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fasteitSolver_t self = *solverPointer;

    // cleanup
    fasteit_forward_solver_release(&self->forwardSolver);
    fasteit_inverse_solver_release(&self->inverseSolver);
    linalgcu_matrix_release(&self->dGamma);
    linalgcu_matrix_release(&self->gamma);
    linalgcu_matrix_release(&self->measuredVoltage);
    linalgcu_matrix_release(&self->calibrationVoltage);

    // free struct
    free(self);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// pre solve for accurate initial jacobian
linalgcuError_t fasteit_solver_pre_solve(fasteitSolver_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // forward solving a few steps
    error |= fasteit_forward_solver_solve(self->forwardSolver, self->gamma, 1000, handle,
        stream);

    // calc system matrix
    error |= fasteit_inverse_solver_calc_system_matrix(self->inverseSolver,
        self->forwardSolver->jacobian, handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    error |= linalgcu_matrix_copy(self->measuredVoltage, self->forwardSolver->voltage, stream);
    error |= linalgcu_matrix_copy(self->calibrationVoltage, self->forwardSolver->voltage,
        stream);

    return error;
}

// calibrate
linalgcuError_t fasteit_solver_calibrate(fasteitSolver_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // solve forward
    error  = fasteit_forward_solver_solve(self->forwardSolver, self->gamma, 10, handle,
        stream);

    // calc inverse system matrix
    error |= fasteit_inverse_solver_calc_system_matrix(self->inverseSolver,
        self->forwardSolver->jacobian, handle, stream);

    // solve inverse
    error |= fasteit_inverse_solver_solve(self->inverseSolver, self->dGamma,
        self->forwardSolver->jacobian, self->forwardSolver->voltage,
        self->calibrationVoltage, 75, handle, stream);

    // add to gamma
    error |= linalgcu_matrix_add(self->gamma, self->dGamma, stream);

    return error;
}

// solving
linalgcuError_t fasteit_solver_solve(fasteitSolver_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // solve
    error |= fasteit_inverse_solver_solve(self->inverseSolver,
        self->dGamma, self->forwardSolver->jacobian, self->calibrationVoltage,
        self->measuredVoltage, 90, handle, stream);

    return error;
}
