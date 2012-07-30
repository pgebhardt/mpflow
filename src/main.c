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

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1E6;
}

int main(int argc, char* argv[]) {
    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // timeing
    double start = get_time();

    // create cublas handle
    cublasHandle_t handle = NULL;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("Cublas handle loaded... (%f ms)\n", (get_time() - start) * 1E3);

    // create mesh
    fastect_mesh_t mesh;
    error = fastect_mesh_create(&mesh, 0.045, 0.045 / 16.0, NULL);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("Mesh generated with r = %f, d = %f ... (%f ms)\n", mesh->radius, mesh->distance,
        (get_time() - start) * 1E3);

    // create electrodes
    fastect_electrodes_t electrodes;
    error  = fastect_electrodes_create(&electrodes, 36, 0.005f, mesh);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("%d electrodes of width = %f generated... (%f ms)\n", electrodes->count,
        electrodes->size, (get_time() - start) * 1E3);

    // load pattern
    linalgcu_matrix_t drive_pattern, measurment_pattern;
    linalgcu_matrix_load(&drive_pattern, "input/drive_pattern.txt", NULL);
    linalgcu_matrix_load(&measurment_pattern, "input/measurment_pattern.txt", NULL);

    // comment
    printf("Measurment and drive pattern loaded... (%f ms)\n", (get_time() - start) * 1E3);

    // create solver
    fastect_solver_t solver;
    error = fastect_solver_create(&solver, mesh, electrodes, 18, 9,
        drive_pattern, measurment_pattern, handle, NULL);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("Solver created... (%f ms)\n", (get_time() - start) * 1E3);

    // Create image
    fastect_image_t image;
    error = fastect_image_create(&image, 1000, 1000, mesh, NULL);

    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("Image module loaded... (%f ms)\n", (get_time() - start) * 1E3);

    // set sigma
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(solver->applied_solver->grid->sigma, 50E-3, i, 0);
        linalgcu_matrix_set_element(solver->lead_solver->grid->sigma, 50E-3, i, 0);
    }
    linalgcu_matrix_copy_to_device(solver->applied_solver->grid->sigma, LINALGCU_TRUE, NULL);
    linalgcu_matrix_copy_to_device(solver->lead_solver->grid->sigma, LINALGCU_TRUE, NULL);
    fastect_grid_update_system_matrix(solver->applied_solver->grid, NULL);
    fastect_grid_update_system_matrix(solver->lead_solver->grid, NULL);

    // comment
    printf("Set initial sigma = %f ... (%f ms)\n", 50E-3, (get_time() - start) * 1E3);

    // load measured_voltage
    linalgcu_matrix_t measured_voltage;
    linalgcu_matrix_load(&measured_voltage, "input/measured_voltage.txt", NULL);
    linalgcu_matrix_copy(solver->measured_voltage, measured_voltage, LINALGCU_TRUE, NULL);
    linalgcu_matrix_release(&measured_voltage);

    // comment
    printf("Measured voltage loaded... (%f ms)\n", (get_time() - start) * 1E3);

    // pre solve for accurate jacobian
    for (linalgcu_size_t i = 0; i < 100; i++) {
        fastect_solver_forward_solve(solver, handle, NULL);
    }
    cudaDeviceSynchronize();

    // comment
    printf("Pre solving done... (%f ms)\n", (get_time() - start) * 1E3);

    // solve
    for (linalgcu_size_t i = 0; i < 10; i++) {
        fastect_solver_solve(solver, 4, handle, NULL);
    }

    // comment
    printf("Solving of 50 frames done... (%f ms)\n", (get_time() - start) * 1E3);

    // calc image
    cudaDeviceSynchronize();
    fastect_image_calc_sigma(image, solver->applied_solver->grid->sigma, NULL);
    cudaDeviceSynchronize();
    linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("output/image.txt", image->image);
    system("python src/script.py reconstructed");

    // comment
    printf("Image created... (%f ms)\n", (get_time() - start) * 1E3);

    // cleanup
    fastect_solver_release(&solver);
    fastect_image_release(&image);
    linalgcu_matrix_release(&drive_pattern);
    linalgcu_matrix_release(&measurment_pattern);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
};
