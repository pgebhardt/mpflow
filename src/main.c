// ert
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
#include "mesh.h"
#include "basis.h"
#include "electrodes.h"
#include "grid.h"
#include "conjugate.h"
#include "image.h"

void print_matrix(linalgcu_matrix_t matrix) {
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

    // create handle
    cublasHandle_t handle = NULL;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return EXIT_SUCCESS;
    }

    // create mesh
    ert_mesh_t mesh;
    error  = ert_mesh_create(&mesh, 0.045, 0.045 / 16.0);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // copy matrices to device
    linalgcu_matrix_copy_to_device(mesh->vertices, LINALGCU_FALSE);
    linalgcu_matrix_copy_to_device(mesh->elements, LINALGCU_TRUE);

    // create electrodes
    ert_electrodes_t electrodes;
    error = ert_electrodes_create(&electrodes, 36, 0.005, mesh);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create grid
    ert_grid_t grid = NULL;
    error  = ert_grid_create(&grid, mesh, handle);
    error |= ert_grid_init_exitation_matrix(grid, electrodes);

    // check success
    if (error != LINALGCU_SUCCESS) {
        printf("Grid error!\n");
        return EXIT_FAILURE;
    }

    // create conjugate solver
    ert_conjugate_solver_t solver = NULL;
    error = ert_conjugate_solver_create(&solver, grid->system_matrix,
        mesh->vertex_count, handle);

    // check success
    if (error != LINALGCU_SUCCESS) {
        printf("solver error!\n");
        return EXIT_FAILURE;
    }

    // Create image
    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh);
    linalgcu_matrix_copy_to_device(image->elements, LINALGCU_FALSE);
    linalgcu_matrix_copy_to_device(image->image, LINALGCU_TRUE);

    // create matrices
    linalgcu_matrix_t f, phi, current;
    linalgcu_matrix_create(&phi, mesh->vertex_count, 1);
    linalgcu_matrix_create(&f, mesh->vertex_count, 1);
    linalgcu_matrix_create(&current, 36, 1);

    // set current
    linalgcu_matrix_set_element(current, 0.02, 1, 0);
    linalgcu_matrix_set_element(current, -0.02, 3, 0);
    linalgcu_matrix_copy_to_device(current, LINALGCU_TRUE);

    // calc f
    linalgcu_matrix_multiply(f, grid->exitation_matrix, current, handle);
    linalgcu_matrix_multiply(f, grid->exitation_matrix, current, handle);

    // get start time
    struct timeval tv;
    cudaStreamSynchronize(NULL);
    gettimeofday(&tv, NULL);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // solve
    error = ert_conjugate_solver_solve(solver, phi, f, 100, handle);

    // get end time
    cudaStreamSynchronize(NULL);
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // print frames per second
    printf("Forward: frames per second: %f\n", (10.0f / 18.0f) /  (end - start));

    if (error != LINALGCU_SUCCESS) {
        printf("Conjugate solving error!\n");
        return EXIT_FAILURE;
    }

    // calc image
    char buffer[1024];
    ert_image_calc_phi(image, phi);
    cudaStreamSynchronize(NULL);
    linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE);
    linalgcu_matrix_save("output/image.txt", image->image);
    sprintf(buffer, "python src/script.py %d", 0);
    system(buffer);

    // cleanup
    linalgcu_matrix_release(&current);
    linalgcu_matrix_release(&f);
    linalgcu_matrix_release(&phi);
    ert_image_release(&image);
    ert_conjugate_solver_release(&solver);
    ert_grid_release(&grid);
    ert_electrodes_release(&electrodes);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
};
