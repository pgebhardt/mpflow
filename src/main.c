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
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->applied_solver->grid->sigma, 50E-3, i, 0);
        linalgcu_matrix_set_element(solver->lead_solver->grid->sigma, 50E-3, i, 0);
    }
    linalgcu_matrix_copy_to_device(solver->applied_solver->grid->sigma, LINALGCU_TRUE, NULL);
    linalgcu_matrix_copy_to_device(solver->lead_solver->grid->sigma, LINALGCU_TRUE, NULL);
    fastect_grid_update_system_matrix(solver->applied_solver->grid, NULL);
    fastect_grid_update_system_matrix(solver->lead_solver->grid, NULL);

    // Create image
    fastect_image_t image;
    fastect_image_create(&image, 1000, 1000, mesh, NULL);
    linalgcu_matrix_copy_to_device(image->elements, LINALGCU_FALSE, NULL);
    linalgcu_matrix_copy_to_device(image->image, LINALGCU_TRUE, NULL);

    for (linalgcu_size_t i = 0; i < 100; i++) {
        fastect_solver_forward_solve(solver, handle, NULL);
    }

    // get start time
    struct timeval tv;
    cudaDeviceSynchronize();
    gettimeofday(&tv, NULL);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    for (linalgcu_size_t i = 0; i < 1; i++) {
        fastect_solver_solve(solver, 4, handle, NULL);
    }

    // get end time
    cudaDeviceSynchronize();
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    printf("Time per frame: %f\n", (end - start) / 5.0);

    // dummy_matrix
    linalgcu_matrix_s dummy_matrix;
    dummy_matrix.host_data = NULL;
    dummy_matrix.device_data = NULL;
    dummy_matrix.size_m = solver->applied_solver->phi->size_m;
    dummy_matrix.size_n = 1;

    // calc images
    /*char buffer[1024];
    for (linalgcu_size_t i = 0; i < solver->applied_solver->count; i++) {
        // copy current phi to vector
        dummy_matrix.device_data =
            &solver->applied_solver->phi->device_data[i * solver->applied_solver->phi->size_m];

        // calc image
        fastect_image_calc_phi(image, &dummy_matrix, NULL);
        cudaDeviceSynchronize();
        linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE, NULL);
        linalgcu_matrix_save("output/image.txt", image->image);
        sprintf(buffer, "python src/script.py %d", i);
        system(buffer);
    }*/

    // calc sigma image
    fastect_image_calc_sigma(image, solver->applied_solver->grid->sigma, NULL);
    cudaDeviceSynchronize();
    linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("output/image.txt", image->image);
    system("python src/script.py 100");

    // cleanup
    fastect_solver_release(&solver);
    fastect_image_release(&image);
    linalgcu_matrix_release(&drive_pattern);
    linalgcu_matrix_release(&measurment_pattern);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
};
