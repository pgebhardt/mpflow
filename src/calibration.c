// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create calibration_solver
linalgcu_error_t fastect_calibration_solver_create(fastect_calibration_solver_t* solverPointer,
    linalgcu_matrix_t jacobian, linalgcu_matrix_data_t regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (jacobian == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastect_calibration_solver_t solver = malloc(sizeof(fastect_calibration_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->conjugate_solver = NULL;
    solver->dVoltage = NULL;
    solver->zeros = NULL;
    solver->excitation = NULL;
    solver->systemMatrix = NULL;
    solver->regularization = NULL;
    solver->regularizationFactor = regularizationFactor;

    // create matrices
    error  = linalgcu_matrix_create(&solver->dVoltage, jacobian->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->dGamma, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->zeros, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->excitation, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->systemMatrix, jacobian->columns,
        jacobian->columns, stream);
    error |= linalgcu_matrix_unity(&solver->regularization, jacobian->columns, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_calibration_solver_release(&solver);

        return error;
    }

    // create conjugate solver
    error = fastect_conjugate_solver_create(&solver->conjugate_solver,
        jacobian->columns, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_calibration_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t fastect_calibration_solver_release(
    fastect_calibration_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_calibration_solver_t solver = *solverPointer;

    // cleanup
    fastect_conjugate_solver_release(&solver->conjugate_solver);
    linalgcu_matrix_release(&solver->dVoltage);
    linalgcu_matrix_release(&solver->dGamma);
    linalgcu_matrix_release(&solver->zeros);
    linalgcu_matrix_release(&solver->excitation);
    linalgcu_matrix_release(&solver->systemMatrix);
    linalgcu_matrix_release(&solver->regularization);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// calc system matrix
linalgcu_error_t fastect_calibration_solver_calc_system_matrix(
    fastect_calibration_solver_t solver, linalgcu_matrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // cublas coeficients
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, solver->systemMatrix->rows,
        solver->systemMatrix->columns, jacobian->rows, &alpha, jacobian->deviceData,
        jacobian->rows, jacobian->deviceData, jacobian->rows, &beta,
        solver->systemMatrix->deviceData, solver->systemMatrix->rows) != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // regularization: L = Jt * J
    // calc regularization
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, solver->systemMatrix->columns,
        solver->systemMatrix->rows, solver->systemMatrix->columns, &alpha,
        solver->systemMatrix->deviceData, solver->systemMatrix->rows,
        solver->systemMatrix->deviceData, solver->systemMatrix->rows,
        &beta, solver->regularization->deviceData, solver->regularization->rows)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc systemMatrix
    if (cublasSaxpy(handle, solver->systemMatrix->rows * solver->systemMatrix->columns,
        &solver->regularizationFactor, solver->regularization->deviceData, 1,
        solver->systemMatrix->deviceData, 1)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// calc excitation
linalgcu_error_t fastect_calibration_solver_calc_excitation(fastect_calibration_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix to turn matrix to column vector
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.rows = solver->dVoltage->rows;
    dummy_matrix.columns = solver->dVoltage->columns;
    dummy_matrix.hostData = NULL;

    // calc deltaVoltage = mv - cv
    dummy_matrix.deviceData = calculatedVoltage->deviceData;
    linalgcu_matrix_copy(solver->dVoltage, &dummy_matrix, LINALGCU_FALSE, stream);
    linalgcu_matrix_scalar_multiply(solver->dVoltage, -1.0f, stream);

    dummy_matrix.deviceData = measuredVoltage->deviceData;
    linalgcu_matrix_add(solver->dVoltage, &dummy_matrix, stream);

    // calc excitation
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
        jacobian->deviceData, jacobian->rows, solver->dVoltage->deviceData, 1, &beta,
        solver->excitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// calibration
linalgcu_error_t fastect_calibration_solver_calibrate(fastect_calibration_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, linalgcu_matrix_t gamma,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // reset dGamma
    error  = linalgcu_matrix_copy(solver->dGamma, solver->zeros, LINALGCU_FALSE, stream);

    // calc system matrix
    error |= fastect_calibration_solver_calc_system_matrix(solver, jacobian, handle, stream);

    // calc excitation
    error |= fastect_calibration_solver_calc_excitation(solver, jacobian, calculatedVoltage,
        measuredVoltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->systemMatrix, solver->dGamma, solver->excitation, steps, handle, stream);

    // add to gamma
    error |= linalgcu_matrix_add(gamma, solver->dGamma, stream);

    return error;
}
