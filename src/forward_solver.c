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
#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include "mesh.h"
#include "basis.h"
#include "electrodes.h"
#include "grid.h"
#include "conjugate.h"
#include "forward.h"

int main(int argc, char* argv[]) {
    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // create cublas handle
    cublasHandle_t handle = NULL;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create mesh
    ert_mesh_t mesh;
    error = ert_mesh_create(&mesh, 0.045, 0.045 / 16.0, NULL);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // copy matrices to device
    linalgcu_matrix_copy_to_device(mesh->vertices, LINALGCU_FALSE, NULL);
    linalgcu_matrix_copy_to_device(mesh->elements, LINALGCU_TRUE, NULL);

    // create electrodes
    ert_electrodes_t electrodes;
    error  = ert_electrodes_create(&electrodes, 36, 0.005f, mesh);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // load pattern
    linalgcu_matrix_t drive_pattern, measurment_pattern;
    linalgcu_matrix_load(&drive_pattern, "input/drive_pattern.txt", NULL);
    linalgcu_matrix_load(&measurment_pattern, "input/measurment_pattern.txt", NULL);
    linalgcu_matrix_copy_to_device(drive_pattern, LINALGCU_TRUE, NULL);
    linalgcu_matrix_copy_to_device(measurment_pattern, LINALGCU_TRUE, NULL);

    // create solver
    ert_forward_solver_t solver;
    error = ert_forward_solver_create(&solver, mesh, electrodes, 18, drive_pattern,
        measurment_pattern, handle, NULL);

    // check success
    if (error != LINALGCU_SUCCESS) {
        printf("Solver erstellen ging nicht!\n");
        return EXIT_FAILURE;
    }

    // set sigma
    linalgcu_matrix_data_t id, x, y;
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->grid->sigma, 50E-3, i, 0);
    }

    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 0);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.005f) * (x - 0.005f) + (y - 0.005f) * (y - 0.005f) > 0.02 * 0.02) {
            continue;
        }

        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 1);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.005f) * (x - 0.005f) + (y - 0.005f) * (y - 0.005f) > 0.02 * 0.02) {
            continue;
        }

        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 2);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.005f) * (x - 0.005f) + (y - 0.005f) * (y - 0.005f) > 0.02 * 0.02) {
            continue;
        }

        // set sigma
        linalgcu_matrix_set_element(solver->grid->sigma, 1E-3, i, 0);
    }
    linalgcu_matrix_copy_to_device(solver->grid->sigma, LINALGCU_TRUE, NULL);
    ert_grid_update_system_matrix(solver->grid, NULL);

    // solve
    for (linalgcu_size_t i = 0; i < 100; i++) {
        ert_forward_solver_solve(solver, handle, NULL);
    }

    // calc voltage
    linalgcu_matrix_t voltage;
    linalgcu_matrix_create(&voltage, measurment_pattern->size_n,
        drive_pattern->size_n, NULL);
    linalgcu_matrix_multiply(voltage, solver->voltage_calculation, solver->phi, handle, NULL);
    cudaStreamSynchronize(NULL);
    linalgcu_matrix_copy_to_host(voltage, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("input/measured_voltage.txt", voltage);
    linalgcu_matrix_release(&voltage);

    // cleanup
    ert_forward_solver_release(&solver);
    linalgcu_matrix_release(&drive_pattern);
    linalgcu_matrix_release(&measurment_pattern);
    cublasDestroy(handle);

    printf("Forward solving done!\n");

    return EXIT_SUCCESS;
};
