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
    linalg_matrix_data_t radius, linalg_matrix_data_t distance) {
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
    error = linalg_matrix_create(&mesh->vertices, (linalg_size_t)(mesh->radius * mesh->radius /
        (0.25 * distance * distance)), 2);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return LINALG_ERROR;
    }

    // create vertices
    linalg_matrix_data_t x = -mesh->radius;
    linalg_matrix_data_t y = -mesh->radius;

    for (int i = 0; i < mesh->vertices->size_x; i++) {
        // check current data set
        if (x * x + y * y <= mesh->radius) {
            linalg_matrix_set_element(mesh->vertices, x, i, 0);
            linalg_matrix_set_element(mesh->vertices, y, i, 1);
            mesh->vortex_count++;
        }
        else {
            i--;
        }

        // calc next vertex
        x += distance;
        if (x > mesh->radius) {
            x -= 2.0 * mesh->radius + distance / 2.0;
            y += sqrt(0.75) * distance;
        }
        if (y >= mesh->radius) {
            break;
        }
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
