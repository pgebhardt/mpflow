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

    return LINALG_SUCCESS;
}
