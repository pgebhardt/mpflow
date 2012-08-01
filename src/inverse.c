// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create inverse_solver
linalgcu_error_t fastect_inverse_solver_create(fastect_inverse_solver_t* solverPointer,
    linalgcu_matrix_t jacobian, linalgcu_matrix_data_t lambda,
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
    solver->jacobian = jacobian;
    solver->dU = NULL;
    solver->dSigma = NULL;
    solver->f = NULL;
    solver->A = NULL;
    solver->regularization = NULL;
    solver->lambda = lambda;

    // create matrices
    error |= linalgcu_matrix_create(&solver->dU, solver->jacobian->size_m, 1, stream);
    error |= linalgcu_matrix_create(&solver->dSigma, solver->jacobian->size_n, 1, stream);
    error |= linalgcu_matrix_create(&solver->f, solver->jacobian->size_n, 1, stream);
    error |= linalgcu_matrix_create(&solver->A, solver->jacobian->size_n,
        solver->jacobian->size_n, stream);
    error |= linalgcu_matrix_unity(&solver->regularization, solver->jacobian->size_n,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_inverse_solver_release(&solver);

        return error;
    }

    // create conjugate solver
    error = fastect_conjugate_solver_create(&solver->conjugate_solver,
        solver->jacobian->size_n, handle, stream);

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
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // cublas coeficients
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, solver->A->size_m, solver->A->size_n,
        solver->jacobian->size_m, &alpha, solver->jacobian->device_data,
        solver->jacobian->size_m, solver->jacobian->device_data, solver->jacobian->size_m,
        &beta, solver->A->device_data, solver->A->size_m) != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc regularization
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, solver->A->size_n, solver->A->size_m,
        solver->A->size_n, &alpha, solver->A->device_data, solver->A->size_m,
        solver->A->device_data, solver->A->size_m,
        &beta, solver->regularization->device_data, solver->regularization->size_m)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc A
    if (cublasSaxpy(handle, solver->A->size_m * solver->A->size_n, &solver->lambda,
        solver->regularization->device_data, 1, solver->A->device_data, 1)
        != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// calc excitation
linalgcu_error_t fastect_inverse_solver_calc_excitation(fastect_inverse_solver_t solver,
    linalgcu_matrix_t calculated_voltage, linalgcu_matrix_t measured_voltage,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (calculated_voltage == NULL) || (measured_voltage == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // reset dU
    cudaMemset(solver->dU, 0,
        sizeof(linalgcu_matrix_data_t) * solver->dU->size_m * solver->dU->size_n);

    // calc dU
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.size_m = solver->dU->size_m;
    dummy_matrix.size_n = solver->dU->size_n;
    dummy_matrix.host_data = NULL;
    dummy_matrix.device_data = calculated_voltage->device_data;
    linalgcu_matrix_add(solver->dU, &dummy_matrix, handle, stream);
    linalgcu_matrix_scalar_multiply(solver->dU, -1.0f, handle, stream);
    dummy_matrix.device_data = measured_voltage->device_data;
    linalgcu_matrix_add(solver->dU, &dummy_matrix, handle, stream);

    // calc f
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, solver->jacobian->size_m, solver->jacobian->size_n,
        &alpha, solver->jacobian->device_data, solver->jacobian->size_m,
        solver->dU->device_data, 1, &beta, solver->f->device_data, 1) != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve(fastect_inverse_solver_t solver,
    linalgcu_matrix_t calculated_voltage, linalgcu_matrix_t measured_voltage,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (calculated_voltage == NULL) || (measured_voltage == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // reset dU, dSigma
    cudaMemset(solver->dSigma, 0,
        sizeof(linalgcu_matrix_data_t) * solver->dSigma->size_m * solver->dSigma->size_n);

    // calc excitation
    error  = fastect_inverse_solver_calc_excitation(solver, calculated_voltage, measured_voltage,
        handle, stream);

    // calc system matrix
    error |= fastect_inverse_solver_calc_system_matrix(solver, handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->A, solver->dSigma, solver->f, 75, handle, stream);

    return error;
}

// linear inverse solving
linalgcu_error_t fastect_inverse_solver_solve_linear(fastect_inverse_solver_t solver,
    linalgcu_matrix_t calculated_voltage, linalgcu_matrix_t measured_voltage,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (calculated_voltage == NULL) || (measured_voltage == NULL) ||
        (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // calc excitation
    error  = fastect_inverse_solver_calc_excitation(solver, calculated_voltage, measured_voltage,
        handle, stream);

    // solve system
    error |= fastect_conjugate_solver_solve(solver->conjugate_solver,
        solver->A, solver->dSigma, solver->f, 75, handle, stream);

    return error;
}
