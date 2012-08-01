// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t drive_count,
    linalgcu_size_t measurment_count, linalgcu_matrix_t drive_pattern,
    linalgcu_matrix_t measurment_pattern, linalgcu_matrix_data_t sigma_ref,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drive_pattern == NULL) || (measurment_pattern == NULL) || (handle == NULL)) {
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
    solver->applied_solver = NULL;
    solver->lead_solver = NULL;
    solver->inverse_solver = NULL;
    solver->mesh = mesh;
    solver->electrodes = electrodes;
    solver->jacobian = NULL;
    solver->voltage_calculation = NULL;
    solver->calculated_voltage = NULL;
    solver->measured_voltage = NULL;
    solver->sigma_ref = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&solver->jacobian,
        measurment_pattern->size_n * drive_pattern->size_n,
        mesh->element_count, stream);
    error |= linalgcu_matrix_create(&solver->voltage_calculation,
        measurment_pattern->size_n, solver->mesh->vertex_count, stream);
    error |= linalgcu_matrix_create(&solver->calculated_voltage,
        measurment_pattern->size_n, drive_pattern->size_n, stream);
    error |= linalgcu_matrix_create(&solver->measured_voltage,
        measurment_count, drive_count, stream);
    error |= linalgcu_matrix_create(&solver->sigma_ref,
        solver->mesh->element_count, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // create solver
    error  = fastect_forward_solver_create(&solver->applied_solver, solver->mesh,
        solver->electrodes, drive_count, drive_pattern, handle, stream);
    error |= fastect_forward_solver_create(&solver->lead_solver, solver->mesh,
        solver->electrodes, measurment_count, measurment_pattern, handle, stream);
    error |= fastect_inverse_solver_create(&solver->inverse_solver,
        solver->jacobian, 5.0f, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // calc voltage calculation matrix
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurment_pattern->size_n,
        solver->applied_solver->grid->excitation_matrix->size_m,
        measurment_pattern->size_m, &alpha, measurment_pattern->device_data,
        measurment_pattern->size_m,
        solver->applied_solver->grid->excitation_matrix->device_data,
        solver->applied_solver->grid->excitation_matrix->size_m,
        &beta, solver->voltage_calculation->device_data, solver->voltage_calculation->size_m)
        != CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // init sigma to sigma ref
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->applied_solver->grid->sigma, sigma_ref, i, 0);
        linalgcu_matrix_set_element(solver->lead_solver->grid->sigma, sigma_ref, i, 0);
        linalgcu_matrix_set_element(solver->sigma_ref, sigma_ref, i, 0);
    }

    // copy sigma to device
    linalgcu_matrix_copy_to_device(solver->applied_solver->grid->sigma, LINALGCU_TRUE, stream);
    linalgcu_matrix_copy_to_device(solver->lead_solver->grid->sigma, LINALGCU_TRUE, stream);
    linalgcu_matrix_copy_to_device(solver->sigma_ref, LINALGCU_TRUE, stream);

    // update forward system matrices
    fastect_grid_update_system_matrix(solver->applied_solver->grid, stream);
    fastect_grid_update_system_matrix(solver->lead_solver->grid, stream);

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// create solver from config file
linalgcu_error_t fastect_solver_from_config(fastect_solver_t* solverPointer,
    config_t* config, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (config == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // setting
    config_setting_t* setting = NULL;

    // read mesh config
    setting = config_lookup(config, "mesh");
    if (setting == NULL) {
        return LINALGCU_ERROR;
    }

    // create mesh
    fastect_mesh_t mesh = NULL;
    error = fastect_mesh_create_from_config(&mesh, setting, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // read electrodes config
    setting = config_lookup(config, "electrodes");
    if (setting == NULL) {
        // cleanup
        fastect_mesh_release(&mesh);

        return LINALGCU_ERROR;
    }

    // create electrodes
    fastect_electrodes_t electrodes = NULL;
    error = fastect_electrodes_create_from_config(&electrodes, setting, mesh->radius);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_mesh_release(&mesh);

        return LINALGCU_ERROR;
    }

    // read solver config
    setting = config_lookup(config, "solver");
    if (setting == NULL) {
        // cleanup
        fastect_electrodes_release(&electrodes);
        fastect_mesh_release(&mesh);

        return LINALGCU_ERROR;
    }

    int measurment_count, drive_count;
    const char* measurment_pattern_path;
    const char* drive_pattern_path;
    double sigma_ref = 0.0;
    if (!(config_setting_lookup_int(setting, "measurment_count", &measurment_count) &&
        config_setting_lookup_int(setting, "drive_count", &drive_count) &&
        config_setting_lookup_string(setting, "measurment_pattern", &measurment_pattern_path) &&
        config_setting_lookup_string(setting, "drive_pattern", &drive_pattern_path) &&
        config_setting_lookup_float(setting, "sigma_ref", &sigma_ref))) {
        // cleanup
        fastect_electrodes_release(&electrodes);
        fastect_mesh_release(&mesh);

        return LINALGCU_ERROR;
    }

    // load pattern
    linalgcu_matrix_t drive_pattern, measurment_pattern;
    error  = linalgcu_matrix_load(&drive_pattern, drive_pattern_path, stream);
    error |= linalgcu_matrix_load(&measurment_pattern, measurment_pattern_path, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_electrodes_release(&electrodes);
        fastect_mesh_release(&mesh);

        return error;
    }

    // create solver
    fastect_solver_t solver;
    error = fastect_solver_create(&solver, mesh, electrodes, drive_count, measurment_count,
        drive_pattern, measurment_pattern, sigma_ref, handle, stream);

    // cleanup
    linalgcu_matrix_release(&drive_pattern);
    linalgcu_matrix_release(&measurment_pattern);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_electrodes_release(&electrodes);
        fastect_mesh_release(&mesh);

        return error;
    }

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
    fastect_forward_solver_release(&solver->applied_solver);
    fastect_forward_solver_release(&solver->lead_solver);
    fastect_inverse_solver_release(&solver->inverse_solver);
    fastect_mesh_release(&solver->mesh);
    fastect_electrodes_release(&solver->electrodes);
    linalgcu_matrix_release(&solver->jacobian);
    linalgcu_matrix_release(&solver->voltage_calculation);
    linalgcu_matrix_release(&solver->calculated_voltage);
    linalgcu_matrix_release(&solver->measured_voltage);
    linalgcu_matrix_release(&solver->sigma_ref);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcu_error_t fastect_solver_forward_solve(fastect_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_ERROR;

    // solve drive pattern
    error  = fastect_grid_update_system_matrix(solver->applied_solver->grid, stream);
    error |= fastect_forward_solver_solve(solver->applied_solver, handle, stream);

    // solve measurment pattern
    error |= fastect_grid_update_system_matrix(solver->lead_solver->grid, stream);
    error |= fastect_forward_solver_solve(solver->lead_solver, handle, stream);

    // calc voltage
    error |= linalgcu_matrix_multiply(solver->calculated_voltage,
        solver->voltage_calculation, solver->applied_solver->phi, handle, stream);

    // calc jacobian
    error |= fastect_solver_calc_jacobian(solver, stream);

    return error;
}

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, linalgcu_size_t linear_frames,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // non-linear inverse
    error = fastect_inverse_solver_solve(solver->inverse_solver, solver->calculated_voltage,
        solver->measured_voltage, solver->applied_solver->grid->sigma, solver->sigma_ref, handle, stream);

    // linear inverse
    for (linalgcu_size_t i = 0; i < linear_frames; i++) {
        error |= fastect_inverse_solver_solve_linear(solver->inverse_solver, solver->calculated_voltage,
            solver->measured_voltage, solver->applied_solver->grid->sigma, solver->sigma_ref, handle, stream);
    }

    // add to sigma
    error |= linalgcu_matrix_add(solver->applied_solver->grid->sigma, solver->inverse_solver->dSigma,
        handle, stream);
    error |= linalgcu_matrix_add(solver->lead_solver->grid->sigma, solver->inverse_solver->dSigma,
        handle, stream);

    // forward
    error |= fastect_solver_forward_solve(solver, handle, stream);

    return error;
}
