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
#include <math.h>
#include <linalg/matrix.h>
#include "mesh.h"

linalg_error_t ert_mesh_create(ert_mesh_t* meshPointer,
    linalg_matrix_data_t radius) {
    // check input
    if ((meshPointer == NULL) || (radius <= 0.0)) {
        return LINALG_ERROR;
    }

    // error
    linalg_error_t error = LINALG_SUCCESS;

    // init mesh pointer
    *meshPointer = NULL;

    // create mesg struct
    ert_mesh_t mesh = malloc(sizeof(ert_mesh_s));

    // check success
    if (mesh == NULL) {
        return LINALG_ERROR;
    }

    // init struct
    mesh->radius = radius;
    mesh->vortex_count = 0;
    mesh->vertices = NULL;

    // create vertex memory
    error = linalg_matrix_create(&mesh->vertices, ERT_MESH_MAX_VERTICES, 2);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return LINALG_ERROR;
    }

    // set mesh pointer
    *meshPointer = mesh;

    return LINALG_SUCCESS;
}

linalg_error_t ert_mesh_release(ert_mesh_t* meshPointer) {
    // check input
    if ((meshPointer == NULL) || (*meshPointer == NULL)) {
        return LINALG_ERROR;
    }

    // get mesh
    ert_mesh_t mesh = *meshPointer;

    // cleanup vertices
    linalg_matrix_release(&mesh->vertices);

    // free struct
    free(mesh);

    // set mesh pointer to NULL
    *meshPointer = NULL;

    return LINALG_SUCCESS;
}

linalg_error_t ert_mesh_init(ert_mesh_t mesh, linalg_matrix_t field) {
    // check input
    if ((mesh == NULL) || (field == NULL)) {
        return LINALG_ERROR;
    }

    // initial vortex distance
    linalg_matrix_data_t distance = 0.05;

    // set vortex count
    mesh->vortex_count = (linalg_size_t)(4.0 * mesh->radius * mesh->radius / (distance * distance));

    // generate uniform mesh
    linalg_matrix_set_element(mesh->vertices, 1.0, 0, 0);
    linalg_matrix_set_element(mesh->vertices, 0.0, 0, 1);

    linalg_matrix_data_t rad = mesh->radius;
    linalg_matrix_data_t phi = 0.0;

    linalg_matrix_data_t x, y;
    for (linalg_size_t i = 1; i < mesh->vortex_count; i++) {
        // calc vortex
        x = rad * cos(phi);
        y = rad * sin(phi);

        linalg_matrix_set_element(mesh->vertices, rad * cos(phi), i, 0);
        linalg_matrix_set_element(mesh->vertices, rad * sin(phi), i, 1);

        // check angle
        if (phi >= 2.0 * M_PI) {
            phi -= 2.0 * M_PI;
            rad -= distance;
        }

        // check radius
        if (rad <= 0.0) {
            mesh->vortex_count = i + 1;
            break;
        }

        // go to next point
        phi += acos(1.0 - distance * distance / (2.0 * rad * rad));
    }

    // precondition field
    for (linalg_size_t i = 0; i < field->size_x; i++) {
        for (linalg_size_t j = 0; j < field->size_y; j++) {
            field->data[(i * field->size_y) + j] = -field->data[(i * field->size_y) + j] *
                field->data[(i * field->size_y) + j] - 0.01;
        }
    }
    // calc gradient of field
    linalg_matrix_t gradx, grady;
    linalg_matrix_create(&gradx, field->size_x, field->size_y);
    linalg_matrix_create(&grady, field->size_x, field->size_y);
    linalg_matrix_data_t dx = 2.0 / (linalg_matrix_data_t)(field->size_x - 1);
    linalg_matrix_data_t dy = 2.0 / (linalg_matrix_data_t)(field->size_y - 1);

    for (linalg_size_t i = 1; i < field->size_x - 1; i++) {
        for (linalg_size_t j = 1; j < field->size_y - 1; j++) {
            gradx->data[(i * gradx->size_y) + j] = (field->data[((i + 1) * field->size_y) + j]
                - field->data[((i - 1) * field->size_y) + j]) / (2.0 * dx);
            grady->data[(i * grady->size_y) + j] = (field->data[(i * field->size_y) + j + 1]
                - field->data[(i * field->size_y) + j - 1]) / (2.0 * dy);

        }
    }

    // move vertices
    linalg_size_t a, b;
    for (linalg_size_t j = 0; j < 10; j++) {
        for (linalg_size_t i = 0; i < mesh->vortex_count; i++) {
            // get vortex position
            linalg_matrix_get_element(mesh->vertices, &x, i, 0);
            linalg_matrix_get_element(mesh->vertices, &y, i, 1);

            // calc matrix positions
            a = (linalg_size_t)((x + 1.0) / dx);
            b = (linalg_size_t)((y + 1.0) / dy);

            // move gradient
            x -= 0.1 * gradx->data[(a * gradx->size_y) + b];
            y -= 0.1 * grady->data[(a * grady->size_y) + b];

            // set vortex
            if (x * x + y * y <= 1.0) {
                linalg_matrix_set_element(mesh->vertices, x, i, 0);
                linalg_matrix_set_element(mesh->vertices, y, i, 1);
            }
        }
    }

    return LINALG_SUCCESS;
}
