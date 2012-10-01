// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create solver
linalgcuError_t fastect_solver_create(fastectSolver_t* solverPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuSize_t numHarmonics,
    linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrix_t measurmentPattern, linalgcuMatrix_t drivePattern,
    linalgcuMatrixData_t sigmaRef, linalgcuMatrixData_t regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream) {
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
    fastectSolver_t self = malloc(sizeof(fastectSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->forwardSolver = NULL;
    self->inverseSolver = NULL;
    self->dGamma = NULL;
    self->gamma = NULL;
    self->jacobian = NULL;
    self->calculatedVoltage = NULL;
    self->measuredVoltage = NULL;
    self->calibrationVoltage = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&self->dGamma, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&self->gamma, mesh->elementCount, 1, stream);
    error |= linalgcu_matrix_create(&self->jacobian,
        measurmentPattern->columns * drivePattern->columns, mesh->elementCount, stream);
    error |= linalgcu_matrix_create(&self->calculatedVoltage, measurmentPattern->columns,
        drivePattern->columns, stream);
    error |= linalgcu_matrix_create(&self->measuredVoltage, measurmentCount, driveCount,
        stream);
    error |= linalgcu_matrix_create(&self->calibrationVoltage, measurmentCount, driveCount,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&self);

        return error;
    }

    // create solver
    error  = fastect_forward_solver_create(&self->forwardSolver, mesh, electrodes,
        sigmaRef, numHarmonics, driveCount, measurmentCount, drivePattern,
        measurmentPattern, handle, stream);
    error |= fastect_inverse_solver_create(&self->inverseSolver,
        mesh->elementCount, measurmentPattern->columns * drivePattern->columns,
        regularizationFactor, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&self);

        return error;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fastect_solver_release(fastectSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectSolver_t self = *solverPointer;

    // cleanup
    fastect_forward_solver_release(&self->forwardSolver);
    fastect_inverse_solver_release(&self->inverseSolver);
    linalgcu_matrix_release(&self->dGamma);
    linalgcu_matrix_release(&self->gamma);
    linalgcu_matrix_release(&self->jacobian);
    linalgcu_matrix_release(&self->calculatedVoltage);
    linalgcu_matrix_release(&self->measuredVoltage);
    linalgcu_matrix_release(&self->calibrationVoltage);

    // free struct
    free(self);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// pre solve for accurate initial jacobian
linalgcuError_t fastect_solver_pre_solve(fastectSolver_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // forward solving a few steps
    error |= fastect_forward_solver_solve(self->forwardSolver, self->jacobian,
        self->gamma, self->calculatedVoltage, 1000, handle, stream);

    // calc system matrix
    error |= fastect_inverse_solver_calc_system_matrix(self->inverseSolver,
        self->jacobian, handle, stream);

    // set measuredVoltage and calculatedVoltage to calculatedVoltage
    error |= linalgcu_matrix_copy(self->measuredVoltage, self->calculatedVoltage, stream);
    error |= linalgcu_matrix_copy(self->calibrationVoltage, self->calculatedVoltage, stream);

    return error;
}

// calibrate
linalgcuError_t fastect_solver_calibrate(fastectSolver_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // forward
    error  = fastect_forward_solver_solve(self->forwardSolver, self->jacobian,
        self->gamma, self->calculatedVoltage, 10, handle, stream);

    // calibration
    error |= fastect_inverse_solver_non_linear(self->inverseSolver,
        self->gamma, self->dGamma, self->jacobian, self->calculatedVoltage,
        self->calibrationVoltage, 75, handle, stream);

    return error;
}

// solving
linalgcuError_t fastect_solver_solve(fastectSolver_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // solve
    error |= fastect_inverse_solver_linear(self->inverseSolver,
        self->dGamma, self->jacobian, self->calibrationVoltage, self->measuredVoltage,
        90, handle, stream);

    return error;
}
