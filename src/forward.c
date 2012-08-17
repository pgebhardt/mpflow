// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (count == 0) || (pattern == NULL) || (handle == NULL)) {
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
    solver->conjugate_solver = NULL;
    solver->count = count;
    solver->phi = NULL;
    solver->f = NULL;

    // create grid
    error  = fastect_grid_create(&solver->grid, mesh, handle, stream);

    // init grid excitaion matrix
    error |= fastect_grid_init_exitation_matrix(solver->grid, mesh, electrodes, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->phi, mesh->vertex_count,
        solver->count, stream);
    error |= linalgcu_matrix_create(&solver->f, mesh->vertex_count,
        solver->count, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create conjugate solver
    error = fastect_conjugate_sparse_solver_create(&solver->conjugate_solver,
        mesh->vertex_count, solver->count, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    // Run multiply once more to avoid cublas error
    linalgcu_matrix_multiply(solver->f, solver->grid->excitation_matrix, pattern, handle, stream);

    error = linalgcu_matrix_multiply(solver->f, solver->grid->excitation_matrix, pattern, handle, stream);

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
    fastect_conjugate_sparse_solver_release(&solver->conjugate_solver);
    linalgcu_matrix_release(&solver->phi);
    linalgcu_matrix_release(&solver->f);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // solve for phi
    error = fastect_conjugate_sparse_solver_solve(solver->conjugate_solver,
        solver->grid->system_matrix, solver->phi, solver->f,
        10, handle, stream);

    return error;
}
