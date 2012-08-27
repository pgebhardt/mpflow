// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_solver_config_t config, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (config == NULL) || (handle == NULL)) {
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
    solver->jacobian = NULL;
    solver->voltage_calculation = NULL;
    solver->calculated_voltage = NULL;
    solver->measured_voltage = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&solver->sigma, config->mesh->element_count, 1, stream);
    error |= linalgcu_matrix_create(&solver->jacobian,
        config->measurment_pattern->columns * config->drive_pattern->columns,
        config->mesh->element_count, stream);
    error |= linalgcu_matrix_create(&solver->voltage_calculation,
        config->measurment_pattern->columns, config->mesh->vertex_count, stream);
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
        config->drive_pattern, config->measurment_pattern, handle, stream);
    error |= fastect_inverse_solver_create(&solver->inverse_solver, solver->jacobian,
        config->regularization_factor, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // calc voltage calculation matrix
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, config->measurment_pattern->columns,
        solver->forward_solver->grid->excitation_matrix->rows,
        config->measurment_pattern->rows, &alpha, config->measurment_pattern->device_data,
        config->measurment_pattern->rows,
        solver->forward_solver->grid->excitation_matrix->device_data,
        solver->forward_solver->grid->excitation_matrix->rows,
        &beta, solver->voltage_calculation->device_data, solver->voltage_calculation->rows)
        != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // init sigma to sigma_0
    for (linalgcu_size_t i = 0; i < config->mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->sigma, config->sigma_0, i, 0);
    }

    // copy sigma to device
    linalgcu_matrix_copy_to_device(solver->sigma, LINALGCU_TRUE, stream);

    // update forward system matrices
    fastect_grid_update_system_matrix(solver->forward_solver->grid, solver->sigma, stream);

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
    linalgcu_matrix_release(&solver->sigma);
    linalgcu_matrix_release(&solver->jacobian);
    linalgcu_matrix_release(&solver->voltage_calculation);
    linalgcu_matrix_release(&solver->calculated_voltage);
    linalgcu_matrix_release(&solver->measured_voltage);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // forward
    error  = fastect_forward_solver_solve(solver->forward_solver, solver->sigma,
        solver->jacobian, 10, handle, stream);

    // calc voltage
    error |= linalgcu_matrix_multiply(solver->calculated_voltage, solver->voltage_calculation,
        solver->forward_solver->drive_phi, handle, stream);

    // inverse
    error |= fastect_inverse_solver_solve(solver->inverse_solver, solver->jacobian,
        solver->calculated_voltage, solver->measured_voltage, solver->sigma, 75, handle,
        stream);

    return error;
}
