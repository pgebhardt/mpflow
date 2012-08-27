// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t drive_count,
    linalgcu_size_t measurment_count, linalgcu_matrix_t drive_pattern,
    linalgcu_matrix_t measurment_pattern, cublasHandle_t handle, cudaStream_t stream) {
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
    fastect_forward_solver_t solver = malloc(sizeof(fastect_forward_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->grid = NULL;
    solver->drive_solver = NULL;
    solver->measurment_solver = NULL;
    solver->drive_phi = NULL;
    solver->measurment_phi = NULL;
    solver->drive_f = NULL;
    solver->measurment_f = NULL;

    // create grid
    error = fastect_grid_create(&solver->grid, mesh, electrodes, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create conjugate solver
    error  = fastect_conjugate_sparse_solver_create(&solver->drive_solver,
        mesh->vertex_count, drive_count, handle, stream);
    error |= fastect_conjugate_sparse_solver_create(&solver->measurment_solver,
        mesh->vertex_count, measurment_count, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->drive_phi, mesh->vertex_count,
        drive_count, stream);
    error |= linalgcu_matrix_create(&solver->measurment_phi, mesh->vertex_count,
        measurment_count, stream);
    error |= linalgcu_matrix_create(&solver->drive_f, mesh->vertex_count,
        drive_count, stream);
    error |= linalgcu_matrix_create(&solver->measurment_f, mesh->vertex_count,
        measurment_count, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    // Run multiply once more to avoid cublas error
    linalgcu_matrix_multiply(solver->drive_f, solver->grid->excitation_matrix, drive_pattern,
        handle, stream);
    error  = linalgcu_matrix_multiply(solver->drive_f, solver->grid->excitation_matrix,
        drive_pattern, handle, stream);
    linalgcu_matrix_multiply(solver->measurment_f, solver->grid->excitation_matrix,
        measurment_pattern, handle, stream);
    error |= linalgcu_matrix_multiply(solver->measurment_f, solver->grid->excitation_matrix,
        measurment_pattern, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t fastect_forward_solver_release(fastect_forward_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_forward_solver_t solver = *solverPointer;

    // cleanup
    fastect_grid_release(&solver->grid);
    fastect_conjugate_sparse_solver_release(&solver->drive_solver);
    fastect_conjugate_sparse_solver_release(&solver->measurment_solver);
    linalgcu_matrix_release(&solver->drive_phi);
    linalgcu_matrix_release(&solver->measurment_phi);
    linalgcu_matrix_release(&solver->drive_f);
    linalgcu_matrix_release(&solver->measurment_f);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    linalgcu_matrix_t sigma, linalgcu_matrix_t jacobian, linalgcu_size_t steps,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (sigma == NULL) || (jacobian == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // update system matrix
    error  = fastect_grid_update_system_matrix(solver->grid, sigma, stream);

    // solve for drive phi
    error |= fastect_conjugate_sparse_solver_solve(solver->drive_solver,
        solver->grid->system_matrix, solver->drive_phi, solver->drive_f,
        steps, handle, stream);

    // solve for measurment phi
    error |= fastect_conjugate_sparse_solver_solve(solver->measurment_solver,
        solver->grid->system_matrix, solver->measurment_phi, solver->measurment_f,
        steps, handle, stream);

    // calc jacobian
    error |= fastect_forward_solver_calc_jacobian(solver, jacobian, stream);

    return error;
}
