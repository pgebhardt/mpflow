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
    mesh->vertices = malloc(ERT_MESH_MAX_VERTICES * sizeof(ert_mesh_vortex_s));

    // check success
    if (mesh->vertices == NULL) {
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

    // free vertices
    free(mesh->vertices);

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

    return LINALG_SUCCESS;
}
