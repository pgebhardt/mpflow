// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include "../include/fastect.h"

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
    printf("Cuda initialized... (%f ms)\n", (get_time() - start) * 1E3);

    // load config file
    config_t config;
    config_init(&config);

    if (!config_read_file(&config, "input/config.cfg")) {
        return EXIT_FAILURE;
    }

    // create solvert from config file
    fastect_solver_t solver;
    error = fastect_solver_from_config(&solver, &config, handle, NULL);

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
    double frame_start = get_time();
    printf("Pre solving done... (%f ms)\n", (frame_start - start) * 1E3);

    // solve
    for (linalgcu_size_t i = 0; i < 10; i++) {
        fastect_solver_solve(solver, 4, handle, NULL);
    }
    cudaDeviceSynchronize();

    // comment
    double frame_end = get_time();
    printf("Solving of 50 frames done... (%f ms)\n", (frame_end - start) * 1E3);
    printf("%f frames per second...\n", 50.0f / (frame_end - frame_start));

    // calc image
    fastect_image_calc_sigma(image, solver->applied_solver->grid->sigma, NULL);
    cudaDeviceSynchronize();
    linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("output/image.txt", image->image);
    system("python script.py reconstructed");
    system("rm -rf output/image.txt");

    // comment
    printf("Image created... (%f ms)\n", (get_time() - start) * 1E3);

    // cleanup
    fastect_solver_release(&solver);
    fastect_image_release(&image);
    cublasDestroy(handle);
    config_destroy(&config);

    return EXIT_SUCCESS;
};
