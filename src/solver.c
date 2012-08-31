// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_solver_config_t config, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (config == NULL)) {
        return LINALGCU_ERROR;
    }

    // check config
    if ((config->drive_pattern == NULL) || (config->measurment_pattern == NULL) ||
        (config->mesh == NULL) || (config->electrodes == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    fastect_solver_t solver = malloc(sizeof(fastect_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->forward_solver = NULL;
    solver->inverse_solver = NULL;
    solver->dSigma = NULL;
    solver->sigma_ref = NULL;
    solver->jacobian = NULL;
    solver->calculated_voltage = NULL;
    solver->measured_voltage = NULL;
    solver->cublas_handle = NULL;
    solver->linear_mode = config->linear_mode;

    // create cublas handle
    if (cublasCreate(&solver->cublas_handle) != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->dSigma, config->mesh->element_count, 1, stream);
    error |= linalgcu_matrix_create(&solver->sigma_ref, config->mesh->element_count, 1, stream);
    error |= linalgcu_matrix_create(&solver->jacobian,
        config->measurment_pattern->columns * config->drive_pattern->columns,
        config->mesh->element_count, stream);
    error |= linalgcu_matrix_create(&solver->calculated_voltage,
        config->measurment_pattern->columns, config->drive_pattern->columns, stream);
    error |= linalgcu_matrix_create(&solver->measured_voltage,
        config->measurment_count, config->drive_count, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // create solver
    error  = fastect_forward_solver_create(&solver->forward_solver, config->mesh,
        config->electrodes, config->drive_count, config->measurment_count,
        config->drive_pattern, config->measurment_pattern, solver->cublas_handle, stream);
    error |= fastect_inverse_solver_create(&solver->inverse_solver, solver->jacobian,
        config->regularization_factor, solver->cublas_handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // init sigma to sigma_0
    for (linalgcu_size_t i = 0; i < config->mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->sigma_ref, config->sigma_0, i, 0);
    }

    // copy sigma to device
    linalgcu_matrix_copy_to_device(solver->sigma_ref, LINALGCU_TRUE, stream);

    // update forward system matrices
    fastect_grid_update_system_matrix(solver->forward_solver->grid, solver->sigma_ref, stream);

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t fastect_solver_release(fastect_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_solver_t solver = *solverPointer;

    // cleanup
    fastect_forward_solver_release(&solver->forward_solver);
    fastect_inverse_solver_release(&solver->inverse_solver);
    linalgcu_matrix_release(&solver->dSigma);
    linalgcu_matrix_release(&solver->sigma_ref);
    linalgcu_matrix_release(&solver->jacobian);
    linalgcu_matrix_release(&solver->calculated_voltage);
    linalgcu_matrix_release(&solver->measured_voltage);
    cublasDestroy(solver->cublas_handle);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// pre solve for accurate initial jacobian
linalgcu_error_t fastect_solver_pre_solve(fastect_solver_t solver, cudaStream_t stream) {
    // check input
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // forward solving a few steps
    error |= fastect_forward_solver_solve(solver->forward_solver, solver->sigma_ref,
        solver->jacobian, solver->calculated_voltage, 1000, solver->cublas_handle, stream);

    // calc system matrix
    error |= fastect_inverse_solver_calc_system_matrix(solver->inverse_solver, solver->jacobian,
        solver->cublas_handle, stream);

    return error;
}

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, cudaStream_t stream) {
    // check input
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    if (solver->linear_mode == LINALGCU_TRUE) {
        // inverse
        error |= fastect_inverse_solver_solve_linear(solver->inverse_solver,
            solver->jacobian, solver->calculated_voltage, solver->measured_voltage,
            solver->dSigma, 90, solver->cublas_handle, stream);
    }
    else {
        // forward
        error  = fastect_forward_solver_solve(solver->forward_solver, solver->sigma_ref,
            solver->jacobian, solver->calculated_voltage, 10, solver->cublas_handle, stream);

        // inverse
        error |= fastect_inverse_solver_solve_non_linear(solver->inverse_solver,
            solver->jacobian, solver->calculated_voltage, solver->measured_voltage,
            solver->dSigma, 75, solver->cublas_handle, stream);

        // add to sigma
        error |= linalgcu_matrix_add(solver->sigma_ref, solver->dSigma, solver->cublas_handle, stream);
    }

    return error;
}
