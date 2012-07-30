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
#include "fastect.h"

// create image
linalgcu_error_t fastect_image_create(fastect_image_t* imagePointer, linalgcu_size_t size_x,
    linalgcu_size_t size_y, fastect_mesh_t mesh, cudaStream_t stream) {
    // check input
    if ((imagePointer == NULL) || (mesh == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init image pointer
    *imagePointer = NULL;

    // create image struct
    fastect_image_t image = malloc(sizeof(fastect_image_s));

    // check success
    if (image == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    image->elements = NULL;
    image->image = NULL;
    image->mesh = mesh;

    // create matrices
    error  = linalgcu_matrix_create(&image->elements, 18, mesh->element_count, stream);
    error += linalgcu_matrix_create(&image->image, size_x, size_y, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_image_release(&image);

        return error;
    }

    // fill elements
    for (linalgcu_size_t i = 0; i < image->image->size_m; i++) {
        for (linalgcu_size_t j = 0; j < image->image->size_n; j++) {
            image->image->host_data[i + j * image->image->size_m] = NAN;
        }
    }

    // fill elements matrix
    linalgcu_matrix_data_t x[3], y[3], id[3];
    fastect_basis_t basis[3];

    for (linalgcu_size_t k = 0; k < mesh->element_count;k++) {
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

        // set matrix elements
        for (linalgcu_size_t i = 0; i < 3; i++) {
            // ids
            linalgcu_matrix_set_element(image->elements, id[i], i, k);

            // coordinates
            linalgcu_matrix_set_element(image->elements, x[i], 3 + 2 * i, k);
            linalgcu_matrix_set_element(image->elements, y[i], 4 + 2 * i, k);

            // basis coefficients
            linalgcu_matrix_set_element(image->elements, basis[i]->coefficients[0],
                9 + 3 * i, k);
            linalgcu_matrix_set_element(image->elements, basis[i]->coefficients[1],
                10 + 3 * i, k);
            linalgcu_matrix_set_element(image->elements, basis[i]->coefficients[2],
                11 + 3 * i, k);
        }

        // cleanup
        fastect_basis_release(&basis[0]);
        fastect_basis_release(&basis[1]);
        fastect_basis_release(&basis[2]);
    }

    // copy matrices to device
    linalgcu_matrix_copy_to_device(image->elements, LINALGCU_TRUE, stream);
    linalgcu_matrix_copy_to_device(image->image, LINALGCU_TRUE, stream);

    // set image pointer
    *imagePointer = image;

    return LINALGCU_SUCCESS;
}

// release image
linalgcu_error_t fastect_image_release(fastect_image_t* imagePointer) {
    // check input
    if ((imagePointer == NULL) || (*imagePointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get image
    fastect_image_t image = *imagePointer;

    // cleanup
    linalgcu_matrix_release(&image->elements);
    linalgcu_matrix_release(&image->image);

    // free struct
    free(image);

    // set image pointer to NULL
    *imagePointer = NULL;

    return LINALGCU_SUCCESS;
}
