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
#include "fastect.h"

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
    solver->inverse_solver = NULL;
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
    error |= fastect_conjugate_solver_create(&solver->inverse_solver,
        solver->mesh->element_count, handle, stream);

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
    fastect_conjugate_solver_release(&solver->inverse_solver);
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

// inverse solving
linalgcu_error_t fastect_solver_inverse_solve(fastect_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // create matrices
    linalgcu_matrix_t dU, dSigma, f, A, temp;
    error  = linalgcu_matrix_create(&dU, solver->calculated_voltage->size_m *
        solver->calculated_voltage->size_n, 1, stream);
    error |= linalgcu_matrix_create(&dSigma, solver->applied_solver->grid->sigma->size_m,
        1, stream);
    error |= linalgcu_matrix_create(&f, solver->applied_solver->grid->sigma->size_m,
        1, stream);
    error |= linalgcu_matrix_create(&A, solver->jacobian->size_n,
        solver->jacobian->size_n, stream);
    error |= linalgcu_matrix_create(&temp, solver->jacobian->size_n,
        solver->jacobian->size_n, stream);

    // calc dU
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.size_m = dU->size_m;
    dummy_matrix.size_n = dU->size_n;
    dummy_matrix.host_data = NULL;
    dummy_matrix.device_data = solver->measured_voltage->device_data;
    linalgcu_matrix_add(dU, &dummy_matrix, handle, stream);

    linalgcu_matrix_scalar_multiply(solver->calculated_voltage, -1.0f, handle, stream);
    dummy_matrix.device_data = solver->calculated_voltage->device_data;
    linalgcu_matrix_add(dU, &dummy_matrix, handle, stream);

    // calc f
    linalgcu_matrix_data_t alpha = 1.0f, beta = 0.0f;
    cublasSgemv(handle, CUBLAS_OP_T, solver->jacobian->size_m, solver->jacobian->size_n,
        &alpha, solver->jacobian->device_data, solver->jacobian->size_m,
        dU->device_data, 1, &beta, f->device_data, 1);

    // calc A
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, A->size_m, A->size_n, solver->jacobian->size_m,
        &alpha, solver->jacobian->device_data, solver->jacobian->size_m,
        solver->jacobian->device_data, solver->jacobian->size_m,
        &beta, A->device_data, A->size_m);

    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->size_n, A->size_m, A->size_n,
        &alpha, A->device_data, A->size_m,
        A->device_data, A->size_m,
        &beta, temp->device_data, temp->size_m);

    alpha = 5.0;
    cublasSaxpy(handle, A->size_m * A->size_n, &alpha, temp->device_data, 1,
        A->device_data, 1);

    // solve system
    fastect_conjugate_solver_solve(solver->inverse_solver,
        A, dSigma, f, 100, handle, stream);

    // add to sigma
    linalgcu_matrix_add(solver->applied_solver->grid->sigma, dSigma, handle, stream);
    linalgcu_matrix_add(solver->lead_solver->grid->sigma, dSigma, handle, stream);

    // cleanup
    linalgcu_matrix_release(&dU);
    linalgcu_matrix_release(&dSigma);
    linalgcu_matrix_release(&f);
    linalgcu_matrix_release(&A);
    linalgcu_matrix_release(&temp);

    return LINALGCU_SUCCESS;
}
