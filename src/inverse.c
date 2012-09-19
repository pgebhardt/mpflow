// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create inverse_solver
linalgcuError_t fastect_inverse_solver_create(fastectInverseSolver_t* solverPointer,
    linalgcuMatrix_t systemMatrix, linalgcuMatrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (systemMatrix == NULL) || (jacobian == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastectInverseSolver_t solver = malloc(sizeof(fastectInverseSolver_s));

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
linalgcuError_t fastect_inverse_solver_release(fastectInverseSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectInverseSolver_t solver = *solverPointer;

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
linalgcuError_t fastect_inverse_solver_calc_excitation(fastectInverseSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix to turn matrix to column vector
    linalgcuMatrix_s dummy_matrix;
    dummy_matrix.rows = solver->deltaVoltage->rows;
    dummy_matrix.columns = solver->deltaVoltage->columns;
    dummy_matrix.hostData = NULL;

    // calc dU = mv - cv
    dummy_matrix.deviceData = calculatedVoltage->deviceData;
    linalgcu_matrix_copy(solver->deltaVoltage, &dummy_matrix, stream);
    linalgcu_matrix_scalar_multiply(solver->deltaVoltage, -1.0f, stream);

    dummy_matrix.deviceData = measuredVoltage->deviceData;
    linalgcu_matrix_add(solver->deltaVoltage, &dummy_matrix, stream);

    // set cublas stream
    cublasSetStream(handle, stream);

    // calc excitation
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
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
linalgcuError_t fastect_inverse_solver_solve(fastectInverseSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, linalgcuMatrix_t gamma,
    linalgcuSize_t steps, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculatedVoltage == NULL) ||
        (measuredVoltage == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // reset dSigma
    error  = linalgcu_matrix_copy(gamma, solver->zeros, stream);

    // calc excitation
    error |= fastect_inverse_solver_calc_excitation(solver, jacobian, calculatedVoltage,
        measuredVoltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->systemMatrix, gamma, solver->excitation, steps, handle, stream);

    return error;
}
