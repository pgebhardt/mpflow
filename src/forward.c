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
#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include "basis.h"
#include "mesh.h"
#include "electrodes.h"
#include "grid.h"
#include "conjugate.h"
#include "forward.h"

static void print_matrix(linalgcu_matrix_t matrix) {
    if (matrix == NULL) {
        return;
    }

    // value memory
    linalgcu_matrix_data_t value = 0.0;

    for (linalgcu_size_t i = 0; i < matrix->size_m; i++) {
        for (linalgcu_size_t j = 0; j < matrix->size_n; j++) {
            // get value
            linalgcu_matrix_get_element(matrix, &value, i, j);

            printf("%f, ", value);
        }
        printf("\n");
    }
}

// create forward_solver
linalgcu_error_t ert_forward_solver_create(ert_forward_solver_t* solverPointer,
    ert_mesh_t mesh, ert_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (pattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    ert_forward_solver_t solver = malloc(sizeof(ert_forward_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->grid = NULL;
    solver->conjugate_solver = NULL;
    solver->electrodes = electrodes;
    solver->count = count;
    solver->pattern = pattern;
    solver->phi = NULL;
    solver->temp = NULL;
    solver->f = NULL;

    // create grids
    error  = ert_grid_create(&solver->grid, mesh, handle, stream);
    error |= ert_grid_init_exitation_matrix(solver->grid, solver->electrodes, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        ert_forward_solver_release(&solver);

        return error;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->phi, mesh->vertex_count,
        solver->count, stream);
    error |= linalgcu_matrix_create(&solver->temp, mesh->vertex_count, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        ert_forward_solver_release(&solver);

        return error;
    }

    // copy matrices to device
    linalgcu_matrix_copy_to_device(solver->phi, LINALGCU_FALSE, stream);
    linalgcu_matrix_copy_to_device(solver->temp, LINALGCU_TRUE, stream);

    // create f matrix storage
    solver->f = malloc(sizeof(linalgcu_matrix_t) * solver->count);

    // check success
    if (solver->f == NULL) {
        // cleanup
        ert_forward_solver_release(&solver);

        return LINALGCU_ERROR;
    }

    // create conjugate solver
    solver->conjugate_solver = malloc(sizeof(ert_conjugate_solver_s) * 2);

    error  = ert_conjugate_solver_create(&solver->conjugate_solver,
        solver->grid->system_matrix, mesh->vertex_count,
        handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        ert_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    error = ert_forward_solver_calc_excitaion(solver, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        ert_forward_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t ert_forward_solver_release(ert_forward_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    ert_forward_solver_t solver = *solverPointer;

    // cleanup
    ert_grid_release(&solver->grid);
    ert_conjugate_solver_release(&solver->conjugate_solver);
    ert_electrodes_release(&solver->electrodes);
    linalgcu_matrix_release(&solver->pattern);
    linalgcu_matrix_release(&solver->phi);
    linalgcu_matrix_release(&solver->temp);

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
linalgcu_error_t ert_forward_solver_calc_excitaion(ert_forward_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // dummy matrix struct
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.host_data = NULL;
    dummy_matrix.device_data = NULL;
    dummy_matrix.size_m = solver->pattern->size_m;
    dummy_matrix.size_n = 1;

    // create drive pattern
    for (linalgcu_size_t i = 0; i < solver->count; i++) {
        // create matrix
        linalgcu_matrix_create(&solver->f[i], solver->grid->mesh->vertex_count, 1, stream);

        // get current pattern
        dummy_matrix.device_data = &solver->pattern->device_data[i * solver->pattern->size_m];

        // calc f
        linalgcu_matrix_multiply(solver->f[i], solver->grid->excitation_matrix,
            &dummy_matrix, handle, stream);
    }

    return LINALGCU_SUCCESS;
}

// forward solving
linalgcu_error_t ert_forward_solver_solve(ert_forward_solver_t solver,
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
        ert_conjugate_solver_solve(solver->conjugate_solver,
            &dummy_matrix, solver->f[i], 10, handle, stream);
    }

    return LINALGCU_SUCCESS;
}
