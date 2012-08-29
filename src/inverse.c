// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include "../include/fastect.h"

// create inverse_solver
linalgcu_error_t fastect_inverse_solver_create(fastect_inverse_solver_t* solverPointer,
    linalgcu_matrix_t jacobian, linalgcu_matrix_data_t regularization_factor,
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
    fastect_inverse_solver_t solver = malloc(sizeof(fastect_inverse_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->conjugate_solver = NULL;
    solver->dU = NULL;
    solver->dSigma = NULL;
    solver->zeros = NULL;
    solver->f = NULL;
    solver->A = NULL;
    solver->regularization = NULL;
    solver->regularization_factor = regularization_factor;

    // create matrices
    error  = linalgcu_matrix_create(&solver->dU, jacobian->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->dSigma, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->zeros, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->f, jacobian->columns, 1, stream);
    error |= linalgcu_matrix_create(&solver->A, jacobian->columns, jacobian->columns, stream);
    error |= linalgcu_matrix_unity(&solver->regularization, jacobian->columns, stream);

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
    linalgcu_matrix_release(&solver->dU);
    linalgcu_matrix_release(&solver->dSigma);
    linalgcu_matrix_release(&solver->zeros);
    linalgcu_matrix_release(&solver->f);
    linalgcu_matrix_release(&solver->A);
    linalgcu_matrix_release(&solver->regularization);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// calc system matrix
linalgcu_error_t fastect_inverse_solver_calc_system_matrix(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // cublas coeficients
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, solver->A->rows, solver->A->columns,
        jacobian->rows, &alpha, jacobian->device_data, jacobian->rows, jacobian->device_data,
        jacobian->rows, &beta, solver->A->device_data, solver->A->rows)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // regularization: L = Jt * J
    // calc regularization
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, solver->A->columns, solver->A->rows,
        solver->A->columns, &alpha, solver->A->device_data, solver->A->rows,
        solver->A->device_data, solver->A->rows,
        &beta, solver->regularization->device_data, solver->regularization->rows)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc A
    if (cublasSaxpy(handle, solver->A->rows * solver->A->columns,
        &solver->regularization_factor, solver->regularization->device_data, 1,
        solver->A->device_data, 1)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// calc excitation
linalgcu_error_t fastect_inverse_solver_calc_excitation(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculated_voltage,
    linalgcu_matrix_t measured_voltage, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculated_voltage == NULL) ||
        (measured_voltage == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix to turn matrix to column vector
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.rows = solver->dU->rows;
    dummy_matrix.columns = solver->dU->columns;
    dummy_matrix.host_data = NULL;

    // calc dU = mv - cv
    dummy_matrix.device_data = calculated_voltage->device_data;
    linalgcu_matrix_copy(solver->dU, &dummy_matrix, LINALGCU_FALSE, stream);
    linalgcu_matrix_scalar_multiply(solver->dU, -1.0f, handle, stream);

    dummy_matrix.device_data = measured_voltage->device_data;
    linalgcu_matrix_add(solver->dU, &dummy_matrix, handle, stream);

    // calc f
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
        jacobian->device_data, jacobian->rows, solver->dU->device_data, 1, &beta,
        solver->f->device_data, 1) != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve_non_linear(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculated_voltage,
    linalgcu_matrix_t measured_voltage, linalgcu_matrix_t sigma,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculated_voltage == NULL) ||
        (measured_voltage == NULL) || (sigma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // reset dSigma
    error  = linalgcu_matrix_copy(solver->dSigma, solver->zeros, LINALGCU_FALSE, stream);

    // calc system matrix
    error |= fastect_inverse_solver_calc_system_matrix(solver, jacobian, handle, stream);

    // calc excitation
    error |= fastect_inverse_solver_calc_excitation(solver, jacobian, calculated_voltage,
        measured_voltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->A, solver->dSigma, solver->f, steps, handle, stream);

    // add to sigma
    error |= linalgcu_matrix_add(sigma, solver->dSigma, handle, stream);

    return error;
}

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve_linear(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculated_voltage,
    linalgcu_matrix_t measured_voltage, linalgcu_size_t steps, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (jacobian == NULL) || (calculated_voltage == NULL) ||
        (measured_voltage == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // reset dSigma
    error  = linalgcu_matrix_copy(solver->dSigma, solver->zeros, LINALGCU_FALSE, stream);

    // calc excitation
    error |= fastect_inverse_solver_calc_excitation(solver, jacobian, calculated_voltage,
        measured_voltage, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->A, solver->dSigma, solver->f, steps, handle, stream);

    return error;
}
