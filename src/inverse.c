// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create inverse_solver
linalgcu_error_t fastect_inverse_solver_create(fastect_inverse_solver_t* solverPointer,
    linalgcu_matrix_t systemMatrix, linalgcu_matrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (systemMatrix == NULL) || (jacobian == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastect_inverse_solver_t solver = malloc(sizeof(fastect_inverse_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->conjugate_solver = NULL;
    solver->deltaVoltage = NULL;
    solver->zeros = NULL;
    solver->excitation = NULL;
    solver->systemMatrix = systemMatrix;

    // create matrices
    error  = linalgcu_matrix_create(&solver->deltaVoltage, jacobian->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->zeros, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->excitation, jacobian->columns, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_inverse_solver_release(&solver);

        return error;
    }

    // create conjugate solver
    error = fastect_conjugate_solver_create(&solver->conjugate_solver,
        jacobian->columns, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_inverse_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t fastect_inverse_solver_release(fastect_inverse_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_inverse_solver_t solver = *solverPointer;

    // cleanup
    fastect_conjugate_solver_release(&solver->conjugate_solver);
    linalgcu_matrix_release(&solver->deltaVoltage);
    linalgcu_matrix_release(&solver->zeros);
    linalgcu_matrix_release(&solver->excitation);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// calc excitation
linalgcu_error_t fastect_inverse_solver_calc_excitation(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix to turn matrix to column vector
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.rows = solver->deltaVoltage->rows;
    dummy_matrix.columns = solver->deltaVoltage->columns;
    dummy_matrix.hostData = NULL;

    // calc dU = mv - cv
    dummy_matrix.deviceData = calculatedVoltage->deviceData;
    linalgcu_matrix_copy(solver->deltaVoltage, &dummy_matrix, LINALGCU_FALSE, stream);
    linalgcu_matrix_scalar_multiply(solver->deltaVoltage, -1.0f, stream);

    dummy_matrix.deviceData = measuredVoltage->deviceData;
    linalgcu_matrix_add(solver->deltaVoltage, &dummy_matrix, stream);

    // calc excitation
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
        jacobian->deviceData, jacobian->rows, solver->deltaVoltage->deviceData, 1, &beta,
        solver->excitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {
        // try once again
        if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
            jacobian->deviceData, jacobian->rows, solver->deltaVoltage->deviceData, 1, &beta,
            solver->excitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {

            return LINALGCU_ERROR;
        }
    }

    return LINALGCU_SUCCESS;
}

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve(fastect_inverse_solver_t solver,
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

    // reset dSigma
    error  = linalgcu_matrix_copy(gamma, solver->zeros, LINALGCU_FALSE, stream);

    // calc excitation
    error |= fastect_inverse_solver_calc_excitation(solver, jacobian, calculatedVoltage,
        measuredVoltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->systemMatrix, gamma, solver->excitation, steps, handle, stream);

    return error;
}
