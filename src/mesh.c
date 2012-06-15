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
    mesh->vertex_count = 0;
    mesh->element_count = 0;
    mesh->boundary_count = 0;
    mesh->vertices = NULL;
    mesh->elements = NULL;
    mesh->boundary = NULL;

    // create vertex memory
    error = linalg_matrix_create(&mesh->vertices, (linalg_size_t)(mesh->radius * mesh->radius /
        (0.25 * distance * distance)), 2);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return error;
    }

    // create vertex matrix
    linalg_matrix_t vertices;
    error = linalg_matrix_create(&vertices, (linalg_size_t)(2.0 * mesh->radius / distance + 1),
        (linalg_size_t)(2.0 * mesh->radius / distance + 1));

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return error;
    }

    // create vertices
    linalg_matrix_data_t x, y;
    for (linalg_size_t i = 0; i < vertices->size_x; i++) {
        for (linalg_size_t j = 0; j < vertices->size_y; j++) {
            // calc x and y
            x = mesh->radius - distance * (linalg_matrix_data_t)i;
            y = mesh->radius - distance * (linalg_matrix_data_t)j;

            // check point
            if (x * x + y * y <= 1.0) {
                // set vertex
                linalg_matrix_set_element(mesh->vertices, x, mesh->vertex_count, 0);
                linalg_matrix_set_element(mesh->vertices, y, mesh->vertex_count, 1);

                // save vertex id in vertex matrix
                linalg_matrix_set_element(vertices, (linalg_size_t)mesh->vertex_count, i, j);

                mesh->vertex_count++;
            }
            else {
                // invalidate vertex
                linalg_matrix_set_element(vertices, NAN, i, j);
            }
        }
    }

    // create elements
    error = linalg_matrix_create(&mesh->elements, vertices->size_x * vertices->size_y * 2, 3);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        linalg_matrix_release(&vertices);
        ert_mesh_release(&mesh);

        return error;
    }

    linalg_matrix_data_t neighbours[8];
    linalg_matrix_data_t node;

    for (linalg_size_t i = 0; i < vertices->size_x; i++) {
        for (linalg_size_t j = 0; j < vertices->size_y; j++) {
            // get node
            linalg_matrix_get_element(vertices, &node, i, j);

            // check valid node
            if (isnan(node)) {
                continue;
            }

            // get neighbours
            // top
            if (i > 0) {
                linalg_matrix_get_element(vertices, &neighbours[0], i - 1, j);
            }
            else {
                neighbours[0] = NAN;
            }

            // top right
            if ((i > 0) && (j < vertices->size_y - 1)) {
                linalg_matrix_get_element(vertices, &neighbours[1], i - 1, j + 1);
            }
            else {
                neighbours[1] = NAN;
            }

            // right
            if (j < vertices->size_y - 1) {
                linalg_matrix_get_element(vertices, &neighbours[2], i, j + 1);
            }
            else {
                neighbours[2] = NAN;
            }

            // bottom right
            if ((i < vertices->size_x - 1) && (j < vertices->size_y - 1)) {
                linalg_matrix_get_element(vertices, &neighbours[3], i + 1, j + 1);
            }
            else {
                neighbours[3] = NAN;
            }

            // bottom
            if (i < vertices->size_x - 1) {
                linalg_matrix_get_element(vertices, &neighbours[4], i + 1, j);
            }
            else {
                neighbours[4] = NAN;
            }

            // bottom left
            if ((i < vertices->size_x - 1) && (j > 0)) {
                linalg_matrix_get_element(vertices, &neighbours[5], i + 1, j - 1);
            }
            else {
                neighbours[5] = NAN;
            }

            // left
            if (j > 0) {
                linalg_matrix_get_element(vertices, &neighbours[6], i, j -1);
            }
            else {
                neighbours[6] = NAN;
            }

            // top left
            if ((i > 0) && (j > 0)) {
                linalg_matrix_get_element(vertices, &neighbours[7], i - 1, j - 1);
            }
            else {
                neighbours[7] = NAN;
            }

            // create elements bottom right
            if (!isnan(neighbours[3])) {
                if (!isnan(neighbours[2])) {
                    linalg_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                    linalg_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->element_count, 1);
                    linalg_matrix_set_element(mesh->elements, neighbours[2],
                        mesh->element_count, 2);

                    mesh->element_count++;
                }

                if (!isnan(neighbours[4])) {
                    linalg_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                    linalg_matrix_set_element(mesh->elements, neighbours[4],
                        mesh->element_count, 1);
                    linalg_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->element_count, 2);

                    mesh->element_count++;
                }
            }
            else if (!isnan((neighbours[2])) && (!isnan(neighbours[4]))) {
                linalg_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalg_matrix_set_element(mesh->elements, neighbours[4], mesh->element_count, 1);
                linalg_matrix_set_element(mesh->elements, neighbours[2], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[6])) && (!isnan(neighbours[5]))) {
                linalg_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalg_matrix_set_element(mesh->elements, neighbours[5], mesh->element_count, 1);
                linalg_matrix_set_element(mesh->elements, neighbours[4], mesh->element_count, 2);
                mesh->element_count++;
            }

            /*// create elements top left
            if ((isnan(neighbours[6])) && (!isnan(neighbours[7]))) {
                linalg_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalg_matrix_set_element(mesh->elements, neighbours[0], mesh->element_count, 1);
                linalg_matrix_set_element(mesh->elements, neighbours[7], mesh->element_count, 2);
                mesh->element_count++;
            }*/
        }
    }

    // cleanup
    linalg_matrix_release(&vertices);

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

    // cleanup elements
    linalg_matrix_release(&mesh->elements);

    // cleanup boundary
    linalg_matrix_release(&mesh->boundary);

    // free struct
    free(mesh);

    // set mesh pointer to NULL
    *meshPointer = NULL;

    return LINALG_SUCCESS;
}
