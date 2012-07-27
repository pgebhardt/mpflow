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

linalgcu_error_t set_sigma(linalgcu_matrix_t sigma, fastect_mesh_t mesh, cudaStream_t stream) {
    // set sigma
    linalgcu_matrix_data_t id, x, y;
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(sigma, 50E-3, i, 0);
    }

    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 0);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.005f) * (x - 0.005f) + (y - 0.005f) * (y - 0.005f) > 0.01 * 0.01) {
            continue;
        }

        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 1);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.005f) * (x - 0.005f) + (y - 0.005f) * (y - 0.005f) > 0.01 * 0.01) {
            continue;
        }

        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 2);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.005f) * (x - 0.005f) + (y - 0.005f) * (y - 0.005f) > 0.01 * 0.01) {
            continue;
        }

        // set sigma
        linalgcu_matrix_set_element(sigma, 1E-3, i, 0);
    }

    linalgcu_matrix_copy_to_device(sigma, LINALGCU_TRUE, stream);

    return LINALGCU_SUCCESS;
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
    fastect_mesh_t mesh;
    error = fastect_mesh_create(&mesh, 0.045, 0.045 / 16.0, NULL);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create electrodes
    fastect_electrodes_t electrodes;
    error  = fastect_electrodes_create(&electrodes, 36, 0.005f, mesh);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // load pattern
    linalgcu_matrix_t drive_pattern, measurment_pattern;
    linalgcu_matrix_load(&drive_pattern, "input/drive_pattern.txt", NULL);
    linalgcu_matrix_load(&measurment_pattern, "input/measurment_pattern.txt", NULL);

    // create solver
    fastect_solver_t solver;
    error = fastect_solver_create(&solver, mesh, electrodes, 18, 9,
        drive_pattern, measurment_pattern, handle, NULL);

    // check success
    if (error != LINALGCU_SUCCESS) {
        printf("Solver erstellen ging nicht!\n");
        return EXIT_FAILURE;
    }

    // set sigma
    set_sigma(solver->applied_solver->grid->sigma, mesh, NULL);

    // update system_matrix
    fastect_grid_update_system_matrix(solver->applied_solver->grid, NULL);

    // solve
    for (linalgcu_size_t i = 0; i < 100; i++) {
        fastect_solver_forward_solve(solver, handle, NULL);
    }

    // save voltage
    cudaDeviceSynchronize();
    linalgcu_matrix_copy_to_host(solver->calculated_voltage, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("input/measured_voltage.txt", solver->calculated_voltage);

    // save sigma
    linalgcu_matrix_save("input/sigma.txt", solver->applied_solver->grid->sigma);

    // Create image
    fastect_image_t image;
    fastect_image_create(&image, 1000, 1000, mesh, NULL);
    linalgcu_matrix_copy_to_device(image->elements, LINALGCU_FALSE, NULL);
    linalgcu_matrix_copy_to_device(image->image, LINALGCU_TRUE, NULL);

    // calc sigma image
    fastect_image_calc_sigma(image, solver->applied_solver->grid->sigma, NULL);
    cudaDeviceSynchronize();
    linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("output/image.txt", image->image);
    system("python src/script.py 101");

    // cleanup
    fastect_solver_release(&solver);
    fastect_image_release(&image);
    linalgcu_matrix_release(&drive_pattern);
    linalgcu_matrix_release(&measurment_pattern);
    cublasDestroy(handle);

    printf("Forward solving done!\n");

    return EXIT_SUCCESS;
};
