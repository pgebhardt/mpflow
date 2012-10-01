// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create inverse_solver
linalgcuError_t fastect_inverse_solver_create(fastectInverseSolver_t* solverPointer,
    linalgcuSize_t elementCount, linalgcuSize_t voltageCount,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastectInverseSolver_t self = malloc(sizeof(fastectInverseSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->conjugateSolver = NULL;
    self->dVoltage = NULL;
    self->zeros = NULL;
    self->excitation = NULL;
    self->systemMatrix = NULL;
    self->jacobianSquare = NULL;
    self->regularizationFactor = regularizationFactor;

    // create matrices
    error  = linalgcu_matrix_create(&self->dVoltage, voltageCount, 1, stream);
    error |= linalgcu_matrix_create(&self->zeros, elementCount, 1, stream);
    error |= linalgcu_matrix_create(&self->excitation, elementCount, 1, stream);
    error |= linalgcu_matrix_create(&self->systemMatrix, elementCount,
        elementCount, stream);
    error |= linalgcu_matrix_create(&self->jacobianSquare, elementCount,
        elementCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_inverse_solver_release(&self);

        return error;
    }

    // create conjugate solver
    error = fastect_conjugate_solver_create(&self->conjugateSolver,
        elementCount, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_inverse_solver_release(&self);

        return error;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fastect_inverse_solver_release(
    fastectInverseSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectInverseSolver_t self = *solverPointer;

    // cleanup
    fastect_conjugate_solver_release(&self->conjugateSolver);
    linalgcu_matrix_release(&self->dVoltage);
    linalgcu_matrix_release(&self->zeros);
    linalgcu_matrix_release(&self->excitation);
    linalgcu_matrix_release(&self->systemMatrix);
    linalgcu_matrix_release(&self->jacobianSquare);

    // free struct
    free(self);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// calc system matrix
linalgcuError_t fastect_inverse_solver_calc_system_matrix(
    fastectInverseSolver_t self, linalgcuMatrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (jacobian == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // cublas coeficients
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, self->jacobianSquare->rows,
        self->jacobianSquare->columns, jacobian->rows, &alpha, jacobian->deviceData,
        jacobian->rows, jacobian->deviceData, jacobian->rows, &beta,
        self->jacobianSquare->deviceData, self->jacobianSquare->rows)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // copy jacobianSquare to systemMatrix
    error |= linalgcu_matrix_copy(self->systemMatrix, self->jacobianSquare, stream);

    // add lambda * Jt * J * Jt * J to systemMatrix
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, self->jacobianSquare->columns,
        self->jacobianSquare->rows, self->jacobianSquare->columns,
        &self->regularizationFactor, self->jacobianSquare->deviceData,
        self->jacobianSquare->rows, self->jacobianSquare->deviceData,
        self->jacobianSquare->rows, &alpha, self->systemMatrix->deviceData,
        self->systemMatrix->rows) != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return error;
}

// calc excitation
linalgcuError_t fastect_inverse_solver_calc_excitation(fastectInverseSolver_t self,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix to turn matrix to column vector
    linalgcuMatrix_s dummy_matrix;
    dummy_matrix.rows = self->dVoltage->rows;
    dummy_matrix.columns = self->dVoltage->columns;
    dummy_matrix.hostData = NULL;

    // calc deltaVoltage = mv - cv
    dummy_matrix.deviceData = calculatedVoltage->deviceData;
    linalgcu_matrix_copy(self->dVoltage, &dummy_matrix, stream);
    linalgcu_matrix_scalar_multiply(self->dVoltage, -1.0f, stream);

    dummy_matrix.deviceData = measuredVoltage->deviceData;
    linalgcu_matrix_add(self->dVoltage, &dummy_matrix, stream);

    // set cublas stream
    cublasSetStream(handle, stream);

    // calc excitation
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
        jacobian->deviceData, jacobian->rows, self->dVoltage->deviceData, 1, &beta,
        self->excitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {

        // try once again
        if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
            jacobian->deviceData, jacobian->rows, self->dVoltage->deviceData, 1, &beta,
            self->excitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {
            return LINALGCU_ERROR;
        }
    }

    return LINALGCU_SUCCESS;
}

// inverse solving non linear
linalgcuError_t fastect_inverse_solver_non_linear(fastectInverseSolver_t self,
    linalgcuMatrix_t gamma, linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian,
    linalgcuMatrix_t calculatedVoltage, linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (gamma == NULL) || (dGamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // reset dGamma
    error  = linalgcu_matrix_copy(dGamma, self->zeros, stream);

    // calc system matrix
    error |= fastect_inverse_solver_calc_system_matrix(self, jacobian, handle, stream);

    // calc excitation
    error |= fastect_inverse_solver_calc_excitation(self, jacobian, calculatedVoltage,
        measuredVoltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(self->conjugateSolver,
        self->systemMatrix, dGamma, self->excitation, steps, handle, stream);

    // add to gamma
    error |= linalgcu_matrix_add(gamma, dGamma, stream);

    return error;
}

// inverse solving linear
linalgcuError_t fastect_inverse_solver_linear(fastectInverseSolver_t self,
    linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (dGamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // reset dSigma
    error  = linalgcu_matrix_copy(dGamma, self->zeros, stream);

    // calc excitation
    error |= fastect_inverse_solver_calc_excitation(self, jacobian, calculatedVoltage,
        measuredVoltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(self->conjugateSolver,
        self->jacobianSquare, dGamma, self->excitation, steps, handle, stream);

    return error;
}
