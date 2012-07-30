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

    // create solvert from config file
    fastect_solver_t solver;
    error = fastect_solver_from_config(&solver, "input/config.cfg", handle, NULL);

    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("Solver created... (%f ms)\n", (get_time() - start) * 1E3);

    // Create image
    fastect_image_t image;
    error = fastect_image_create(&image, 1000, 1000, solver->mesh, NULL);

    if (error != LINALGCU_SUCCESS) {
        return EXIT_FAILURE;
    }

    // comment
    printf("Image module loaded... (%f ms)\n", (get_time() - start) * 1E3);

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
    system("rm -rf output/image.txt");

    // comment
    printf("Image created... (%f ms)\n", (get_time() - start) * 1E3);

    // cleanup
    fastect_solver_release(&solver);
    fastect_image_release(&image);
    cublasDestroy(handle);

    return EXIT_SUCCESS;
};
