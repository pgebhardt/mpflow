// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <stdlib.h>
#include <math.h>
#include "fastect.h"

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t drive_pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (drive_pattern == NULL) || (handle == NULL)) {
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

    // create grids
    error  = fastect_grid_create(&solver->grid, mesh, handle, stream);
    error |= fastect_grid_init_exitation_matrix(solver->grid, mesh, electrodes, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create matrices
    error = linalgcu_matrix_create(&solver->phi, mesh->vertex_count,
        solver->count, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create f matrix storage
    solver->f = malloc(sizeof(linalgcu_matrix_t) * solver->count);

    // check success
    if (solver->f == NULL) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // create conjugate solver
    error = fastect_conjugate_solver_create(&solver->conjugate_solver,
        mesh->vertex_count, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    error = fastect_forward_solver_calc_excitaion(solver, mesh, drive_pattern, handle, stream);

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
    fastect_conjugate_solver_release(&solver->conjugate_solver);
    linalgcu_matrix_release(&solver->phi);

    if (solver->f != NULL) {
        for (linalgcu_size_t i = 0; i < solver->count; i++) {
            linalgcu_matrix_release(&solver->f[i]);
        }
        free(solver->f);
    }

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// calc excitaion
linalgcu_error_t fastect_forward_solver_calc_excitaion(fastect_forward_solver_t solver,
    fastect_mesh_t mesh, linalgcu_matrix_t drive_pattern, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (mesh == NULL) || (drive_pattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix struct
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.host_data = NULL;
    dummy_matrix.device_data = NULL;
    dummy_matrix.size_m = drive_pattern->size_m;
    dummy_matrix.size_n = 1;

    // create drive pattern
    for (linalgcu_size_t i = 0; i < solver->count; i++) {
        // create matrix
        linalgcu_matrix_create(&solver->f[i], mesh->vertex_count, 1, stream);

        // get current pattern
        dummy_matrix.device_data = &drive_pattern->device_data[i * drive_pattern->size_m];

        // calc f
        linalgcu_matrix_multiply(solver->f[i], solver->grid->excitation_matrix,
            &dummy_matrix, handle, stream);
    }

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix struct
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.host_data = NULL;
    dummy_matrix.device_data = NULL;
    dummy_matrix.size_m = solver->phi->size_m;
    dummy_matrix.size_n = 1;

    // solve pattern
    for (linalgcu_size_t i = 0; i < solver->count; i++) {
        // copy current applied phi to vector
        dummy_matrix.device_data = &solver->phi->device_data[i * solver->phi->size_m];

        // solve for phi
        fastect_conjugate_solver_solve_sparse(solver->conjugate_solver,
            solver->grid->system_matrix, &dummy_matrix, solver->f[i],
            10, handle, stream);
    }

    return LINALGCU_SUCCESS;
}
