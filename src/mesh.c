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
#include <stdio.h>
#include <math.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include "mesh.h"

linalgcu_matrix_data_t fastect_mesh_angle(linalgcu_matrix_data_t x,
    linalgcu_matrix_data_t y) {
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

int compare_boundary(const void* a, const void* b) {
    linalgcu_matrix_data_t temp = ((linalgcu_matrix_data_t*)a)[1] - ((linalgcu_matrix_data_t*)b)[1];

    if (temp > 0) {
        return 1;
    }
    else if (temp < 0) {
        return -1;
    }
    else {
        return 0;
    }
}

linalgcu_error_t fastect_mesh_create(fastect_mesh_t* meshPointer,
    linalgcu_matrix_data_t radius, linalgcu_matrix_data_t distance,
    cudaStream_t stream) {
    // check input
    if ((meshPointer == NULL) || (radius <= 0.0)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init mesh pointer
    *meshPointer = NULL;

    // create mesg struct
    fastect_mesh_t mesh = malloc(sizeof(fastect_mesh_s));

    // check success
    if (mesh == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    mesh->radius = radius;
    mesh->distance = distance;
    mesh->vertex_count = 0;
    mesh->element_count = 0;
    mesh->boundary_count = 0;
    mesh->vertices = NULL;
    mesh->elements = NULL;
    mesh->boundary = NULL;

    // create vertex memory
    error = linalgcu_matrix_create(&mesh->vertices,
        2 * (linalgcu_size_t)(mesh->radius * mesh->radius /
        (0.25 * mesh->distance * mesh->distance)), 2, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_mesh_release(&mesh);

        return error;
    }

    // create vertex matrix
    linalgcu_matrix_t vertices;
    linalgcu_matrix_data_t r = sqrt(3.0 * mesh->distance * mesh->distance / 4.0);
    error = linalgcu_matrix_create(&vertices,
        (linalgcu_size_t)(3.0 * mesh->radius / mesh->distance + 1),
        (linalgcu_size_t)(3.0 * mesh->radius / mesh->distance + 1), stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_mesh_release(&mesh);

        return error;
    }

    // create vertices
    linalgcu_matrix_data_t x, y;

    for (linalgcu_size_t i = 0; i < vertices->size_m; i++) {
        for (linalgcu_size_t j = 0; j < vertices->size_n; j++) {
            // calc x and y
            x = 1.25 * mesh->radius - mesh->distance * (linalgcu_matrix_data_t)i +
                (linalgcu_matrix_data_t)(j % 2) * 0.5 * mesh->distance;
            y = (int)(mesh->radius / r + 1) * r - r * (linalgcu_matrix_data_t)j;

            // check point
            if (sqrt(x * x + y * y) < mesh->radius - 0.25f * mesh->distance) {
                // set vertex
                linalgcu_matrix_set_element(mesh->vertices, x, mesh->vertex_count, 0);
                linalgcu_matrix_set_element(mesh->vertices, y, mesh->vertex_count, 1);

                // save vertex id in vertex matrix
                linalgcu_matrix_set_element(vertices, (linalgcu_size_t)mesh->vertex_count, i, j);

                mesh->vertex_count++;
            }
            else if (sqrt(x * x + y * y) < mesh->radius + 0.5f * mesh->distance) {
                // calc new x and y
                x = x / sqrt(x * x + y * y) * mesh->radius;
                y = y / sqrt(x * x + y * y) * mesh->radius;

                // set vertex
                linalgcu_matrix_set_element(mesh->vertices, x, mesh->vertex_count, 0);
                linalgcu_matrix_set_element(mesh->vertices, y, mesh->vertex_count, 1);

                // save vertex id in vertex matrix
                linalgcu_matrix_set_element(vertices, (linalgcu_size_t)mesh->vertex_count, i, j);

                mesh->vertex_count++;
            }
            else {
                // invalidate vertex
                linalgcu_matrix_set_element(vertices, NAN, i, j);
            }
        }
    }

    // create boundary matrix
    error = linalgcu_matrix_create(&mesh->boundary,
        (linalgcu_size_t)(3.0 * M_PI * mesh->radius / mesh->distance), 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&vertices);
        fastect_mesh_release(&mesh);

        return error;
    }

    // get boundary vertices
    linalgcu_matrix_data_t boundary[mesh->boundary->size_m * 2];
    linalgcu_matrix_data_t id;
    linalgcu_matrix_data_t angle;

    for (linalgcu_size_t i = 0; i < mesh->vertex_count; i++) {
        // get vertex
        linalgcu_matrix_get_element(mesh->vertices, &x, i, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, i, 1);

        // calc radius and angle
        radius = sqrt(x * x + y * y);
        angle = fastect_mesh_angle(x, y);
        angle += angle < 0.0f ? 2.0 * M_PI : 0.0f;

        // check radius and angle
        if (radius >= mesh->radius - mesh->distance / 4.0f) {
            boundary[mesh->boundary_count * 2 + 0] = (linalgcu_matrix_data_t)i;
            boundary[mesh->boundary_count * 2 + 1] = angle;
            mesh->boundary_count++;
        }
    }

    // sort boundary
    qsort(boundary, mesh->boundary_count, 2 * sizeof(linalgcu_matrix_data_t), compare_boundary);

    // write boundary ids to matrix
    for (linalgcu_size_t i = 0; i < mesh->boundary_count; i++) {
        linalgcu_matrix_set_element(mesh->boundary, boundary[i * 2], i, 0);
    }

    // create elements
    error = linalgcu_matrix_create(&mesh->elements,
        mesh->vertex_count * 6, 3, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&vertices);
        fastect_mesh_release(&mesh);

        return error;
    }

    linalgcu_matrix_data_t neighbours[6];
    linalgcu_matrix_data_t node;

    for (linalgcu_size_t i = 0; i < vertices->size_m; i++) {
        for (linalgcu_size_t j = 0; j < vertices->size_n; j++) {
            // get node
            linalgcu_matrix_get_element(vertices, &node, i, j);

            // check valid node
            if (isnan(node)) {
                continue;
            }

            // get neighbours
            if (j % 2 == 1) {
                // top left
                if ((i > 0) && (j > 0)) {
                    linalgcu_matrix_get_element(vertices, &neighbours[0], i - 1, j - 1);
                }
                else {
                    neighbours[0] = NAN;
                }

                // top right
                if (j > 0) {
                    linalgcu_matrix_get_element(vertices, &neighbours[1], i, j - 1);
                }
                else {
                    neighbours[1] = NAN;
                }

                // right
                if (i < vertices->size_m - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[2], i + 1, j);
                }
                else {
                    neighbours[2] = NAN;
                }

                // bottom right
                if (j < vertices-> size_n - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[3], i, j + 1);
                }
                else {
                    neighbours[3] = NAN;
                }

                // bottom left
                if ((i > 0) && (j < vertices->size_n - 1)) {
                    linalgcu_matrix_get_element(vertices, &neighbours[4], i - 1, j + 1);
                }
                else {
                    neighbours[4] = NAN;
                }

                // left
                if (i > 0) {
                    linalgcu_matrix_get_element(vertices, &neighbours[5], i - 1, j);
                }
                else {
                    neighbours[5] = NAN;
                }
            }
            else {
                // top left
                if (j > 0) {
                    linalgcu_matrix_get_element(vertices, &neighbours[0], i, j - 1);
                }
                else {
                    neighbours[0] = NAN;
                }

                // top right
                if ((i < vertices->size_m - 1) && (j > 0)) {
                    linalgcu_matrix_get_element(vertices, &neighbours[1], i + 1, j - 1);
                }
                else {
                    neighbours[1] = NAN;
                }

                // right
                if (i < vertices->size_m - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[2], i + 1, j);
                }
                else {
                    neighbours[2] = NAN;
                }

                // bottom right
                if ((i < vertices->size_m - 1) && (j < vertices->size_n - 1)) {
                    linalgcu_matrix_get_element(vertices, &neighbours[3], i + 1, j + 1);
                }
                else {
                    neighbours[3] = NAN;
                }

                // bottom left
                if (j < vertices->size_n - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[4], i, j + 1);
                }
                else {
                    neighbours[4] = NAN;
                }

                // left
                if (i > 0) {
                    linalgcu_matrix_get_element(vertices, &neighbours[5], i - 1, j);
                }
                else {
                    neighbours[5] = NAN;
                }
            }

            // create elements bottom right
            if (!isnan(neighbours[3])) {
                if (!isnan(neighbours[2])) {
                    linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->element_count, 1);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[2],
                        mesh->element_count, 2);

                    mesh->element_count++;
                }

                if (!isnan(neighbours[4])) {
                    linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[4],
                        mesh->element_count, 1);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->element_count, 2);

                    mesh->element_count++;
                }
            }

            // create elements bottom left
            if ((isnan(neighbours[5])) && (!isnan(neighbours[4])) && (!isnan(neighbours[0]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[0], mesh->element_count, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[4], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[2])) && (!isnan(neighbours[1])) && (!isnan(neighbours[3]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[3], mesh->element_count, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[1], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[0])) && (!isnan(neighbours[1])) && (!isnan(neighbours[5]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[1], mesh->element_count, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[5], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[1])) && (!isnan(neighbours[2])) && (!isnan(neighbours[0]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[2], mesh->element_count, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[0], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[4])) && (!isnan(neighbours[5])) && (!isnan(neighbours[3]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[5], mesh->element_count, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[3], mesh->element_count, 2);
                mesh->element_count++;
            }

            // create elements bottom left
            if ((isnan(neighbours[3])) && (!isnan(neighbours[4])) && (!isnan(neighbours[2]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->element_count, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[4], mesh->element_count, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[2], mesh->element_count, 2);
                mesh->element_count++;
            }
        }
    }

    // cleanup
    linalgcu_matrix_release(&vertices);

    // set mesh pointer
    *meshPointer = mesh;

    return LINALGCU_SUCCESS;
}

linalgcu_error_t fastect_mesh_release(fastect_mesh_t* meshPointer) {
    // check input
    if ((meshPointer == NULL) || (*meshPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get mesh
    fastect_mesh_t mesh = *meshPointer;

    // cleanup vertices
    linalgcu_matrix_release(&mesh->vertices);

    // cleanup elements
    linalgcu_matrix_release(&mesh->elements);

    // cleanup boundary
    linalgcu_matrix_release(&mesh->boundary);

    // free struct
    free(mesh);

    // set mesh pointer to NULL
    *meshPointer = NULL;

    return LINALGCU_SUCCESS;
}
