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

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t drive_pattern, linalgcu_matrix_t measurment_pattern,
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
    fastect_forward_solver_t solver = malloc(sizeof(fastect_forward_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->grid = NULL;
    solver->conjugate_solver = NULL;
    solver->electrodes = electrodes;
    solver->count = count;
    solver->phi = NULL;
    solver->f = NULL;
    solver->voltage_calculation = NULL;

    // create grids
    error  = fastect_grid_create(&solver->grid, mesh, handle, stream);
    error |= fastect_grid_init_exitation_matrix(solver->grid, solver->electrodes, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->phi, mesh->vertex_count,
        solver->count, stream);
    error |= linalgcu_matrix_create(&solver->voltage_calculation,
        measurment_pattern->size_n, mesh->vertex_count, stream);

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
        solver->grid->system_matrix, mesh->vertex_count,
        handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc excitaion matrices
    error = fastect_forward_solver_calc_excitaion(solver, drive_pattern, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return error;
    }

    // calc voltage calculation matrix
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurment_pattern->size_n,
        solver->grid->excitation_matrix->size_m, measurment_pattern->size_m, &alpha,
        measurment_pattern->device_data, measurment_pattern->size_m,
        solver->grid->excitation_matrix->device_data, solver->grid->excitation_matrix->size_m,
        &beta, solver->voltage_calculation->device_data, solver->voltage_calculation->size_m) !=
        CUBLAS_STATUS_SUCCESS) {
        // cleanup
        fastect_forward_solver_release(&solver);

        return LINALGCU_ERROR;
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
    fastect_electrodes_release(&solver->electrodes);
    linalgcu_matrix_release(&solver->phi);
    linalgcu_matrix_release(&solver->voltage_calculation);

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
    linalgcu_matrix_t drive_pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
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
        linalgcu_matrix_create(&solver->f[i], solver->grid->mesh->vertex_count, 1, stream);

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
        fastect_conjugate_solver_solve(solver->conjugate_solver,
            &dummy_matrix, solver->f[i], 10, handle, stream);
    }

    return LINALGCU_SUCCESS;
}
