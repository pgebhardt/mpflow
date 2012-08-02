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

linalgcu_error_t set_sigma(linalgcu_matrix_t sigma, fastect_mesh_t mesh, cudaStream_t stream) {
    // set sigma
    linalgcu_matrix_data_t id, x, y;
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 0);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.01f) * (x - 0.01f) + y * y > 0.01 * 0.01) {
            continue;
        }

        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 1);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.01f) * (x - 0.01f) + y * y > 0.01 * 0.01) {
            continue;
        }

        // get element
        linalgcu_matrix_get_element(mesh->elements, &id, i, 2);
        linalgcu_matrix_get_element(mesh->vertices, &x, (linalgcu_size_t)id, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, (linalgcu_size_t)id, 1);

        // check element
        if ((x - 0.01f) * (x - 0.01f) + y * y > 0.01 * 0.01) {
            continue;
        }

        // set sigma
        linalgcu_matrix_set_element(sigma, 1E-6, i, 0);
    }

    linalgcu_matrix_copy_to_device(sigma, LINALGCU_TRUE, stream);

    return LINALGCU_SUCCESS;
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

    // set sigma
    set_sigma(solver->applied_solver->grid->sigma, solver->mesh, NULL);
    linalgcu_matrix_copy_to_device(solver->applied_solver->grid->sigma, LINALGCU_TRUE, NULL);
    fastect_grid_update_system_matrix(solver->applied_solver->grid, NULL);

    // comment
    printf("Set sigma... (%f ms)\n", (get_time() - start) * 1E3);

    // solve forward problem several times
    for (linalgcu_size_t i = 0; i < 100; i++) {
        fastect_solver_forward_solve(solver, handle, NULL);
    }
    cudaDeviceSynchronize();

    // comment
    printf("Solving done... (%f ms)\n", (get_time() - start) * 1E3);

    // calc image
    fastect_image_calc_sigma(image, solver->applied_solver->grid->sigma, NULL);
    cudaDeviceSynchronize();
    linalgcu_matrix_copy_to_host(image->image, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("output/image.txt", image->image);
    system("python script.py original");
    system("rm -rf output/image.txt");

    // comment
    printf("Image created... (%f ms)\n", (get_time() - start) * 1E3);

    // save voltage
    linalgcu_matrix_copy_to_host(solver->calculated_voltage, LINALGCU_TRUE, NULL);
    linalgcu_matrix_save("input/measured_voltage.txt", solver->calculated_voltage);

    // comment
    printf("Voltage saved... (%f ms)\n", (get_time() - start) * 1E3);

    // cleanup
    fastect_solver_release(&solver);
    fastect_image_release(&image);
    cublasDestroy(handle);
    config_destroy(&config);

    return EXIT_SUCCESS;
};
