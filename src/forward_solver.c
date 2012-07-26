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
#include <sys/time.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include <actor/actor.h>
#include "mesh.h"
#include "basis.h"
#include "image.h"
#include "electrodes.h"
#include "grid.h"
#include "forward.h"
#include "solver.h"

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
        printf("Mesh erzeugen ging nicht!\n");
        return EXIT_FAILURE;
    }

    // copy matrices to device
    linalgcu_matrix_copy_to_device(mesh->vertices, LINALGCU_TRUE, NULL);
    linalgcu_matrix_copy_to_device(mesh->elements, LINALGCU_TRUE, NULL);

    // create electrodes
    ert_electrodes_t electrodes;
    error = ert_electrodes_create(&electrodes, 36, 0.005, mesh);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create solver
    ert_solver_t solver;
    error = ert_solver_create(&solver, mesh, electrodes, 9, 18,
        program, context, device_id, queue);

    // check success
    if (error != LINALGCU_SUCCESS) {
        printf("Kann keinen Solver erstellen!\n");
        return EXIT_FAILURE;
    }

    // set sigma
    linalgcu_matrix_data_t id;
    linalgcu_matrix_data_t x, y;
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->sigma, 1.0f, i, 0);
    }

    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        // check element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 0);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        if ((x - 0.005) * (x - 0.005) + (y - 0.005) * (y - 0.005) > 0.02 * 0.02) {
            continue;
        }

        linalgcu_matrix_get_element(mesh->elements, &id, i, 1);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        if ((x - 0.005) * (x - 0.005) + (y - 0.005) * (y - 0.005) > 0.02 * 0.02) {
            continue;
        }

        linalgcu_matrix_get_element(mesh->elements, &id, i, 2);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        if ((x - 0.005) * (x - 0.005) + (y - 0.005) * (y - 0.005) > 0.02 * 0.02) {
            continue;
        }

        linalgcu_matrix_set_element(solver->sigma, 0.1f, i, 0);
    }
    linalgcu_matrix_copy_to_device(solver->sigma, queue, LINALGCU_TRUE);
    ert_grid_update_system_matrix(solver->grid, queue);

    // solve
    ert_solver_forward(NULL, solver, program, context, queue);

    // voltage
    linalgcu_matrix_copy_to_host(solver->calculated_voltage, queue, LINALGCU_TRUE);
    linalgcu_matrix_save("input/measured_voltage.txt", solver->calculated_voltage);

    // save sigma
    linalgcu_matrix_copy_to_host(solver->sigma, queue, LINALGCU_TRUE);
    linalgcu_matrix_save("input/sigma.txt", solver->sigma);

    // cleanup
    ert_solver_release(&solver);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
};
