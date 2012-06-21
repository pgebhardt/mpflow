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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <linalgcl/linalgcl.h>
#include "mesh.h"

linalgcl_error_t ert_mesh_create(ert_mesh_t* meshPointer,
    linalgcl_matrix_data_t radius, linalgcl_matrix_data_t distance,
    cl_context context) {
    // check input
    if ((meshPointer == NULL) || (radius <= 0.0)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init mesh pointer
    *meshPointer = NULL;

    // create mesg struct
    ert_mesh_t mesh = malloc(sizeof(ert_mesh_s));

    // check success
    if (mesh == NULL) {
        return LINALGCL_ERROR;
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
    error = linalgcl_matrix_create(&mesh->vertices, context, 2 * (linalgcl_size_t)(mesh->radius * mesh->radius /
        (0.25 * distance * distance)), 2);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return error;
    }

    // create vertex matrix
    linalgcl_matrix_t vertices;
    linalgcl_matrix_data_t r = sqrt(3.0 * distance * distance / 4.0);
    error = linalgcl_matrix_create(&vertices, context,
        (linalgcl_size_t)(3.0 * mesh->radius / distance + 1),
        (linalgcl_size_t)(3.0 * mesh->radius / distance + 1));

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return error;
    }

    // create vertices
    linalgcl_matrix_data_t x, y;

    for (linalgcl_size_t i = 0; i < vertices->size_x; i++) {
        for (linalgcl_size_t j = 0; j < vertices->size_y; j++) {
            // calc x and y
            x = 1.25 * mesh->radius - distance * (linalgcl_matrix_data_t)i +
                (linalgcl_matrix_data_t)(j % 2) * 0.5 * distance;
            y = (int)(mesh->radius / r + 1) * r - r * (linalgcl_matrix_data_t)j;

            // check point
            if (x * x + y * y < mesh->radius - 0.5 * distance) {
                // set vertex
                linalgcl_matrix_set_element(mesh->vertices, x, mesh->vertex_count, 0);
                linalgcl_matrix_set_element(mesh->vertices, y, mesh->vertex_count, 1);

                // save vertex id in vertex matrix
                linalgcl_matrix_set_element(vertices, (linalgcl_size_t)mesh->vertex_count, i, j);

                mesh->vertex_count++;
            }
            else if (x * x + y * y < mesh->radius + distance) {
                // calc new x and y
                x = x / sqrt(x * x + y * y) * mesh->radius;
                y = y / sqrt(x * x + y * y) * mesh->radius;

                // set vertex
                linalgcl_matrix_set_element(mesh->vertices, x, mesh->vertex_count, 0);
                linalgcl_matrix_set_element(mesh->vertices, y, mesh->vertex_count, 1);

                // save vertex id in vertex matrix
                linalgcl_matrix_set_element(vertices, (linalgcl_size_t)mesh->vertex_count, i, j);

                mesh->vertex_count++;
            }
            else {
                // invalidate vertex
                linalgcl_matrix_set_element(vertices, NAN, i, j);
            }
        }
    }

    // create boundary matrix
    error = linalgcl_matrix_create(&mesh->boundary, context,
        (linalgcl_size_t)(3.0 * M_PI * mesh->radius / distance), 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        linalgcl_matrix_release(&vertices);
        ert_mesh_release(&mesh);

        return error;
    }

    // get boundary vertices
    // look from right to left
    linalgcl_matrix_data_t id = NAN;

    for (linalgcl_size_t i = 0; i < vertices->size_x; i++) {
        for (linalgcl_size_t j = 0; j < vertices->size_y; j++) {
            // get vertex id
            linalgcl_matrix_get_element(vertices, &id, i, j);

            // check elements
            if (!isnan(id)) {
                // add new boundary vertex
                linalgcl_matrix_set_element(mesh->boundary, id, mesh->boundary_count, 0);

                // increment boundary count
                mesh->boundary_count++;

                break;
            }
        }
    }

    // look from left to right
    for (int i = vertices->size_x - 1; i >= 0; i--) {
        for (int j = vertices->size_y - 1; j >= 0; j--) {
            // get vertex id
            linalgcl_matrix_get_element(vertices, &id, i, j);

            // check elements
            if ((!isnan(id)) && (id != mesh->boundary->host_data[mesh->boundary_count - 1])) {
                // add new boundary vertex
                linalgcl_matrix_set_element(mesh->boundary, id, mesh->boundary_count, 0);

                // increment boundary count
                mesh->boundary_count++;

                break;
            }
        }
    }

    // create elements
    error = linalgcl_matrix_create(&mesh->elements, context,
        vertices->size_x * vertices->size_y * 2, 3);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        linalgcl_matrix_release(&vertices);
        ert_mesh_release(&mesh);

        return error;
    }

    linalgcl_matrix_data_t neighbours[6];
    linalgcl_matrix_data_t node;

    for (linalgcl_size_t i = 0; i < vertices->size_x; i++) {
        for (linalgcl_size_t j = 0; j < vertices->size_y; j++) {
            // get node
            linalgcl_matrix_get_element(vertices, &node, i, j);

            // check valid node
            if (isnan(node)) {
                continue;
            }

            // get neighbours
            if (j % 2 == 1) {
                // top left
                if ((i > 0) && (j > 0)) {
                    linalgcl_matrix_get_element(vertices, &neighbours[0], i - 1, j - 1);
                }
                else {
                    neighbours[0] = NAN;
                }

                // top right
                if (j > 0) {
                    linalgcl_matrix_get_element(vertices, &neighbours[1], i, j - 1);
                }
                else {
                    neighbours[1] = NAN;
                }

                // right
                if (i < vertices->size_x - 1) {
                    linalgcl_matrix_get_element(vertices, &neighbours[2], i + 1, j);
                }
                else {
                    neighbours[2] = NAN;
                }

                // bottom right
                if (j < vertices-> size_y - 1) {
                    linalgcl_matrix_get_element(vertices, &neighbours[3], i, j + 1);
                }
                else {
                    neighbours[3] = NAN;
                }

                // bottom left
                if ((i > 0) && (j < vertices-> size_y - 1)) {
                    linalgcl_matrix_get_element(vertices, &neighbours[4], i - 1, j + 1);
                }
                else {
                    neighbours[4] = NAN;
                }

                // left
                if (i > 0) {
                    linalgcl_matrix_get_element(vertices, &neighbours[5], i - 1, j);
                }
                else {
                    neighbours[5] = NAN;
                }
            }
            else {
                // top left
                if (j > 0) {
                    linalgcl_matrix_get_element(vertices, &neighbours[0], i, j - 1);
                }
                else {
                    neighbours[0] = NAN;
                }

                // top right
                if ((i < vertices->size_x - 1) && (j > 0)) {
                    linalgcl_matrix_get_element(vertices, &neighbours[1], i + 1, j - 1);
                }
                else {
                    neighbours[1] = NAN;
                }

                // right
                if (i < vertices->size_x - 1) {
                    linalgcl_matrix_get_element(vertices, &neighbours[2], i + 1, j);
                }
                else {
                    neighbours[2] = NAN;
                }

                // bottom right
                if ((i < vertices->size_x - 1) && (j < vertices-> size_y - 1)) {
                    linalgcl_matrix_get_element(vertices, &neighbours[3], i + 1, j + 1);
                }
                else {
                    neighbours[3] = NAN;
                }

                // bottom left
                if (j < vertices-> size_y - 1) {
                    linalgcl_matrix_get_element(vertices, &neighbours[4], i, j + 1);
                }
                else {
                    neighbours[4] = NAN;
                }

                // left
                if (i > 0) {
                    linalgcl_matrix_get_element(vertices, &neighbours[5], i - 1, j);
                }
                else {
                    neighbours[5] = NAN;
                }
            }

            // create elements bottom right
            if (!isnan(neighbours[3])) {
                if (!isnan(neighbours[2])) {
                    linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                    linalgcl_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->element_count, 1);
                    linalgcl_matrix_set_element(mesh->elements, neighbours[2],
                        mesh->element_count, 2);

                    mesh->element_count++;
                }

                if (!isnan(neighbours[4])) {
                    linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                    linalgcl_matrix_set_element(mesh->elements, neighbours[4],
                        mesh->element_count, 1);
                    linalgcl_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->element_count, 2);

                    mesh->element_count++;
                }
            }

            // create elements bottom left
            if ((isnan(neighbours[5])) && (!isnan(neighbours[4])) && (!isnan(neighbours[0]))) {
                linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcl_matrix_set_element(mesh->elements, neighbours[0], mesh->element_count, 1);
                linalgcl_matrix_set_element(mesh->elements, neighbours[4], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[2])) && (!isnan(neighbours[1])) && (!isnan(neighbours[3]))) {
                linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcl_matrix_set_element(mesh->elements, neighbours[3], mesh->element_count, 1);
                linalgcl_matrix_set_element(mesh->elements, neighbours[1], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[0])) && (!isnan(neighbours[1])) && (!isnan(neighbours[5]))) {
                linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcl_matrix_set_element(mesh->elements, neighbours[1], mesh->element_count, 1);
                linalgcl_matrix_set_element(mesh->elements, neighbours[5], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[1])) && (!isnan(neighbours[2])) && (!isnan(neighbours[0]))) {
                linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcl_matrix_set_element(mesh->elements, neighbours[2], mesh->element_count, 1);
                linalgcl_matrix_set_element(mesh->elements, neighbours[0], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[4])) && (!isnan(neighbours[5])) && (!isnan(neighbours[3]))) {
                linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcl_matrix_set_element(mesh->elements, neighbours[5], mesh->element_count, 1);
                linalgcl_matrix_set_element(mesh->elements, neighbours[3], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[3])) && (!isnan(neighbours[4])) && (!isnan(neighbours[2]))) {
                linalgcl_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcl_matrix_set_element(mesh->elements, neighbours[4], mesh->element_count, 1);
                linalgcl_matrix_set_element(mesh->elements, neighbours[2], mesh->element_count, 2);
                mesh->element_count++;
            }
        }
    }

    // cleanup
    linalgcl_matrix_release(&vertices);

    // set mesh pointer
    *meshPointer = mesh;

    return LINALGCL_SUCCESS;
}

linalgcl_error_t ert_mesh_release(ert_mesh_t* meshPointer) {
    // check input
    if ((meshPointer == NULL) || (*meshPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get mesh
    ert_mesh_t mesh = *meshPointer;

    // cleanup vertices
    linalgcl_matrix_release(&mesh->vertices);

    // cleanup elements
    linalgcl_matrix_release(&mesh->elements);

    // cleanup boundary
    linalgcl_matrix_release(&mesh->boundary);

    // free struct
    free(mesh);

    // set mesh pointer to NULL
    *meshPointer = NULL;

    return LINALGCL_SUCCESS;
}
