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
#include "fastect.h"

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
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
    fastect_solver_t solver = malloc(sizeof(fastect_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->applied_solver = NULL;
    solver->lead_solver = NULL;
    solver->mesh = mesh;
    solver->electrodes = electrodes;
    solver->jacobian = NULL;
    solver->voltage_calculation = NULL;
    solver->calculated_voltage = NULL;
    solver->measured_voltage = NULL;

    // create solver
    error  = fastect_forward_solver_create(&solver->applied_solver, solver->mesh,
        solver->electrodes, drive_count, drive_pattern, handle, stream);
    error |= fastect_forward_solver_create(&solver->lead_solver, solver->mesh,
        solver->electrodes, measurment_count, measurment_pattern, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_solver_release(&solver);

        return error;
    }

    // create matrices
    error  = linalgcu_matrix_create(&solver->jacobian,
        measurment_pattern->size_n * drive_pattern->size_n,
        mesh->element_count, stream);
    error |= linalgcu_matrix_create(&solver->voltage_calculation,
        measurment_pattern->size_n, solver->mesh->vertex_count, stream);
    error |= linalgcu_matrix_create(&solver->calculated_voltage,
        measurment_pattern->size_n, drive_pattern->size_n, stream);
    // TODO: measured_voltage should be set by ethernet
    error |= linalgcu_matrix_load(&solver->measured_voltage,
        "input/measured_voltage.txt", stream);

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
    fastect_mesh_release(&solver->mesh);
    fastect_electrodes_release(&solver->electrodes);
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
    error  = fastect_forward_solver_solve(solver->applied_solver, handle, stream);

    // solve measurment pattern
    error |= fastect_forward_solver_solve(solver->lead_solver, handle, stream);

    // calc voltage
    error |= linalgcu_matrix_multiply(solver->calculated_voltage,
        solver->voltage_calculation, solver->applied_solver->phi, handle, stream);

    // calc jacobian
    error |= fastect_solver_calc_jacobian(solver, stream);

    // check error
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    return LINALGCU_SUCCESS;
}
