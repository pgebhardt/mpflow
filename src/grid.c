// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create solver grid
linalgcu_error_t fastect_grid_create(fastect_grid_t* gridPointer,
    fastect_mesh_t mesh, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((gridPointer == NULL) || (mesh == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init grid pointer
    *gridPointer = NULL;

    // create grid struct
    fastect_grid_t grid = malloc(sizeof(fastect_grid_s));

    // check success
    if (grid == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    grid->system_matrix = NULL;
    grid->excitation_matrix = NULL;
    grid->gradient_matrix_sparse = NULL;
    grid->gradient_matrix_transposed_sparse = NULL;
    grid->gradient_matrix_transposed = NULL;
    grid->sigma = NULL;
    grid->area = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&grid->gradient_matrix_transposed,
        mesh->vertex_count, 2 * mesh->element_count, stream);
    error |= linalgcu_matrix_create(&grid->sigma, mesh->element_count, 1, stream);
    error |= linalgcu_matrix_create(&grid->area, mesh->element_count, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // init to uniform sigma
    for (linalgcu_size_t i = 0; i < mesh->element_count; i++) {
        linalgcu_matrix_set_element(grid->sigma, 1.0, i, 0);
    }
    linalgcu_matrix_copy_to_device(grid->sigma, LINALGCU_TRUE, stream);

    // init system matrix
    error = fastect_grid_init_system_matrix(grid, mesh, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_grid_release(&grid);

        return error;
    }

    // set grid pointer
    *gridPointer = grid;

    return LINALGCU_SUCCESS;
}

// release solver grid
linalgcu_error_t fastect_grid_release(fastect_grid_t* gridPointer) {
    // check input
    if ((gridPointer == NULL) || (*gridPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get grid
    fastect_grid_t grid = *gridPointer;

    // cleanup
    linalgcu_sparse_matrix_release(&grid->system_matrix);
    linalgcu_matrix_release(&grid->excitation_matrix);
    linalgcu_matrix_release(&grid->gradient_matrix_transposed);
    linalgcu_sparse_matrix_release(&grid->gradient_matrix_sparse);
    linalgcu_sparse_matrix_release(&grid->gradient_matrix_transposed_sparse);
    linalgcu_matrix_release(&grid->sigma);
    linalgcu_matrix_release(&grid->area);

    // free struct
    free(grid);

    // set grid pointer to NULL
    *gridPointer = NULL;

    return LINALGCU_SUCCESS;
}

// init system matrix
linalgcu_error_t fastect_grid_init_system_matrix(fastect_grid_t grid,
    fastect_mesh_t mesh, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (mesh == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // calc initial system matrix
    // create matrices
    linalgcu_matrix_t system_matrix, gradient_matrix, sigma_matrix;
    error = linalgcu_matrix_create(&system_matrix,
        mesh->vertex_count, mesh->vertex_count, stream);
    error += linalgcu_matrix_create(&gradient_matrix,
        2 * mesh->element_count, mesh->vertex_count, stream);
    error += linalgcu_matrix_unity(&sigma_matrix,
        2 * mesh->element_count, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc gradient matrix
    linalgcu_matrix_data_t x[3], y[3];
    linalgcu_matrix_data_t id[3];
    fastect_basis_t basis[3];

    // init matrices
    for (linalgcu_size_t i = 0; i < gradient_matrix->size_m; i++) {
        for (linalgcu_size_t j = 0; j < gradient_matrix->size_n; j++) {
            linalgcu_matrix_set_element(gradient_matrix, 0.0, i, j);
            linalgcu_matrix_set_element(grid->gradient_matrix_transposed, 0.0, j, i);
        }
    }

    linalgcu_matrix_data_t area;

    for (linalgcu_size_t k = 0; k < mesh->element_count; k++) {
        // get vertices for element
        for (linalgcu_size_t i = 0; i < 3; i++) {
            linalgcu_matrix_get_element(mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(mesh->vertices, &x[i], (linalgcu_size_t)id[i], 0);
            linalgcu_matrix_get_element(mesh->vertices, &y[i], (linalgcu_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        fastect_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        fastect_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        fastect_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        for (linalgcu_size_t i = 0; i < 3; i++) {
            linalgcu_matrix_set_element(gradient_matrix,
                basis[i]->gradient[0], 2 * k, (linalgcu_size_t)id[i]);
            linalgcu_matrix_set_element(gradient_matrix,
                basis[i]->gradient[1], 2 * k + 1, (linalgcu_size_t)id[i]);
            linalgcu_matrix_set_element(grid->gradient_matrix_transposed,
                basis[i]->gradient[0], (linalgcu_size_t)id[i], 2 * k);
            linalgcu_matrix_set_element(grid->gradient_matrix_transposed,
                basis[i]->gradient[1], (linalgcu_size_t)id[i], 2 * k + 1);
        }

        // calc area of element
        area = 0.5 * fabs((x[1] - x[0]) * (y[2] - y[0]) -
            (x[2] - x[0]) * (y[1] - y[0]));

        linalgcu_matrix_set_element(grid->area, area, k, 0);
        linalgcu_matrix_set_element(sigma_matrix, grid->sigma->host_data[k] * area,
            2 * k, 2 * k);
        linalgcu_matrix_set_element(sigma_matrix, grid->sigma->host_data[k] * area,
            2 * k + 1, 2 * k + 1);

        // cleanup
        fastect_basis_release(&basis[0]);
        fastect_basis_release(&basis[1]);
        fastect_basis_release(&basis[2]);
    }

    // copy matrices to device
    error  = linalgcu_matrix_copy_to_device(gradient_matrix, LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(grid->gradient_matrix_transposed,
        LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(sigma_matrix, LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(grid->sigma, LINALGCU_TRUE, stream);
    error |= linalgcu_matrix_copy_to_device(grid->area, LINALGCU_TRUE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&system_matrix);
        linalgcu_matrix_release(&gradient_matrix);
        linalgcu_matrix_release(&sigma_matrix);

        return LINALGCU_ERROR;
    }

    // calc system matrix
    linalgcu_matrix_t temp = NULL;
    error = linalgcu_matrix_create(&temp, mesh->vertex_count,
        2 * mesh->element_count, stream);

    // one prerun cublas to get ready
    linalgcu_matrix_multiply(temp, grid->gradient_matrix_transposed,
        sigma_matrix, handle, stream);

    error |= linalgcu_matrix_multiply(temp, grid->gradient_matrix_transposed,
        sigma_matrix, handle, stream);
    error |= linalgcu_matrix_multiply(system_matrix, temp, gradient_matrix, handle, stream);
    cudaStreamSynchronize(stream);
    linalgcu_matrix_release(&temp);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&system_matrix);

        return LINALGCU_ERROR;
    }

    // create sparse matrices
    error = linalgcu_sparse_matrix_create(&grid->system_matrix, system_matrix, stream);
    error |= linalgcu_sparse_matrix_create(&grid->gradient_matrix_transposed_sparse,
        grid->gradient_matrix_transposed, stream);
    error |= linalgcu_sparse_matrix_create(&grid->gradient_matrix_sparse,
        gradient_matrix, stream);

    // cleanup
    linalgcu_matrix_release(&sigma_matrix);
    linalgcu_matrix_release(&gradient_matrix);
    linalgcu_matrix_release(&system_matrix);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

linalgcu_matrix_data_t fastect_grid_angle(linalgcu_matrix_data_t x, linalgcu_matrix_data_t y) {
    if (x > 0.0f) {
        return atan(y / x);
    }
    else if ((x < 0.0f) && (y >= 0.0f)) {
        return atan(y / x) + M_PI;
    }
    else if ((x < 0.0f) && (y < 0.0f)) {
        return atan(y / x) - M_PI;
    }
    else if ((x == 0.0f) && (y > 0.0f)) {
        return M_PI / 2.0f;
    }
    else if ((x == 0.0f) && (y < 0.0f)) {
        return - M_PI / 2.0f;
    }
    else {
        return 0.0f;
    }
}

linalgcu_matrix_data_t fastect_grid_integrate_basis(linalgcu_matrix_data_t* node,
    linalgcu_matrix_data_t* left, linalgcu_matrix_data_t* right,
    linalgcu_matrix_data_t* start, linalgcu_matrix_data_t* end) {
    // integral
    linalgcu_matrix_data_t integral = 0.0f;

    // calc angle of node
    linalgcu_matrix_data_t angle = fastect_grid_angle(node[0], node[1]);
    linalgcu_matrix_data_t radius = sqrt(start[0] * start[0] + start[1] * start[1]);

    // calc s parameter
    linalgcu_matrix_data_t sleft = radius * (fastect_grid_angle(left[0], left[1]) - angle);
    linalgcu_matrix_data_t sright = radius * (fastect_grid_angle(right[0], right[1]) - angle);
    linalgcu_matrix_data_t sstart = radius * (fastect_grid_angle(start[0], start[1]) - angle);
    linalgcu_matrix_data_t send = radius * (fastect_grid_angle(end[0], end[1]) - angle);

    // correct s parameter
    sleft -= (sleft > sright) && (sleft > 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sright += (sleft > sright) && (sleft < 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sstart -= (sstart > send) && (sstart > 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sstart += (sstart > send) && (sstart < 0.0f) ? radius * 2.0f * M_PI : 0.0f;

    // integrate left triangle
    if ((sstart < 0.0f) && (send > sleft)) {
        if ((send >= 0.0f) && (sstart <= sleft)) {
            integral = -0.5f * sleft;
        }
        else if ((send >= 0.0f) && (sstart > sleft)) {
            integral = -(sstart - 0.5 * sstart * sstart / sleft);
        }
        else if ((send < 0.0f) && (sstart <= sleft)) {
            integral = (send - 0.5 * send * send / sleft) -
                       (sleft - 0.5 * sleft * sleft / sleft);
        }
        else if ((send < 0.0f) && (sstart > sleft)) {
            integral = (send - 0.5 * send * send / sleft) -
                       (sstart - 0.5 * sstart * sstart / sleft);
        }
    }

    // integrate right triangle
    if ((send > 0.0f) && (sright > sstart)) {
        if ((sstart <= 0.0f) && (send >= sright)) {
            integral += 0.5f * sright;
        }
        else if ((sstart <= 0.0f) && (send < sright)) {
            integral += (send - 0.5f * send * send / sright);
        }
        else if ((sstart > 0.0f) && (send >= sright)) {
            integral += (sright - 0.5f * sright * sright / sright) -
                        (sstart - 0.5f * sstart * sstart / sright);
        }
        else if ((sstart > 0.0f) && (send < sright)) {
            integral += (send - 0.5f * send * send / sright) -
                        (sstart - 0.5f * sstart * sstart / sright);
        }
    }

    return integral;
}

// init exitation matrix
linalgcu_error_t fastect_grid_init_exitation_matrix(fastect_grid_t grid,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, cudaStream_t stream) {
    // check input
    if ((grid == NULL) || (mesh == NULL) || (electrodes == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // create exitation_matrix
    error = linalgcu_matrix_create(&grid->excitation_matrix,
        mesh->vertex_count, electrodes->count, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc electrode area
    // linalgcu_matrix_data_t element_area = M_PI * electrodes->size * electrodes->size / 4.0f;
    linalgcu_matrix_data_t element_area = electrodes->size;

    // fill exitation_matrix matrix
    linalgcu_matrix_data_t id[3];
    linalgcu_matrix_data_t node[2], left[2], right[2];

    for (int i = 0; i < mesh->boundary_count; i++) {
        for (int j = 0; j < electrodes->count; j++) {
            // get boundary node id
            linalgcu_matrix_get_element(mesh->boundary, &id[0], i - 1 < 0 ? mesh->boundary_count - 1 : i - 1, 0);
            linalgcu_matrix_get_element(mesh->boundary, &id[1], i, 0);
            linalgcu_matrix_get_element(mesh->boundary, &id[2], (i + 1) % mesh->boundary_count, 0);

            // get coordinates
            linalgcu_matrix_get_element(mesh->vertices, &left[0], (linalgcu_size_t)id[0], 0);
            linalgcu_matrix_get_element(mesh->vertices, &left[1], (linalgcu_size_t)id[0], 1);
            linalgcu_matrix_get_element(mesh->vertices, &node[0], (linalgcu_size_t)id[1], 0);
            linalgcu_matrix_get_element(mesh->vertices, &node[1], (linalgcu_size_t)id[1], 1);
            linalgcu_matrix_get_element(mesh->vertices, &right[0], (linalgcu_size_t)id[2], 0);
            linalgcu_matrix_get_element(mesh->vertices, &right[1], (linalgcu_size_t)id[2], 1);

            // calc element
            linalgcu_matrix_set_element(grid->excitation_matrix,
                fastect_grid_integrate_basis(node, left, right,
                    &electrodes->electrode_start[j * 2], &electrodes->electrode_end[j * 2]) / element_area,
                    (linalgcu_size_t)id[1], j);
        }
    }

    // upload matrix
    linalgcu_matrix_copy_to_device(grid->excitation_matrix, LINALGCU_TRUE, stream);

    return LINALGCU_SUCCESS;
}
