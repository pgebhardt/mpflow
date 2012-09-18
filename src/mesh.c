// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

linalgcuMatrixData_t fastect_mesh_angle(linalgcuMatrixData_t x,
    linalgcuMatrixData_t y) {
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
    linalgcuMatrixData_t temp = ((linalgcuMatrixData_t*)a)[1] - ((linalgcuMatrixData_t*)b)[1];

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

linalgcuError_t fastect_mesh_create(fastectMesh_t* meshPointer,
    linalgcuMatrixData_t radius, linalgcuMatrixData_t distance,
    linalgcuMatrixData_t height, cudaStream_t stream) {
    // check input
    if ((meshPointer == NULL) || (radius <= 0.0f) || (distance <= 0.0f) || (height <= 0.0f)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init mesh pointer
    *meshPointer = NULL;

    // create mesg struct
    fastectMesh_t mesh = malloc(sizeof(fastectMesh_s));

    // check success
    if (mesh == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    mesh->radius = radius;
    mesh->distance = distance;
    mesh->vertexCount = 0;
    mesh->elementCount = 0;
    mesh->boundaryCount = 0;
    mesh->vertices = NULL;
    mesh->elements = NULL;
    mesh->boundary = NULL;
    mesh->height = height;

    // create vertex memory
    error = linalgcu_matrix_create(&mesh->vertices,
        2 * (linalgcuSize_t)(mesh->radius * mesh->radius /
        (0.25 * mesh->distance * mesh->distance)), 2, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_mesh_release(&mesh);

        return error;
    }

    // create vertex matrix
    linalgcuMatrix_t vertices;
    linalgcuMatrixData_t r = sqrt(3.0 * mesh->distance * mesh->distance / 4.0);
    error = linalgcu_matrix_create(&vertices,
        (linalgcuSize_t)(3.0 * mesh->radius / mesh->distance + 1),
        (linalgcuSize_t)(3.0 * mesh->radius / mesh->distance + 1), stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_mesh_release(&mesh);

        return error;
    }

    // create vertices
    linalgcuMatrixData_t x, y;

    for (linalgcuSize_t i = 0; i < vertices->rows; i++) {
        for (linalgcuSize_t j = 0; j < vertices->columns; j++) {
            // calc x and y
            x = 1.25 * mesh->radius - mesh->distance * (linalgcuMatrixData_t)i +
                (linalgcuMatrixData_t)(j % 2) * 0.5 * mesh->distance;
            y = (int)(mesh->radius / r + 1) * r - r * (linalgcuMatrixData_t)j;

            // check point
            if (sqrt(x * x + y * y) < mesh->radius - 0.25f * mesh->distance) {
                // set vertex
                linalgcu_matrix_set_element(mesh->vertices, x, mesh->vertexCount, 0);
                linalgcu_matrix_set_element(mesh->vertices, y, mesh->vertexCount, 1);

                // save vertex id in vertex matrix
                linalgcu_matrix_set_element(vertices, (linalgcuSize_t)mesh->vertexCount, i, j);

                mesh->vertexCount++;
            }
            else if (sqrt(x * x + y * y) < mesh->radius + 0.5f * mesh->distance) {
                // calc new x and y
                x = x / sqrt(x * x + y * y) * mesh->radius;
                y = y / sqrt(x * x + y * y) * mesh->radius;

                // set vertex
                linalgcu_matrix_set_element(mesh->vertices, x, mesh->vertexCount, 0);
                linalgcu_matrix_set_element(mesh->vertices, y, mesh->vertexCount, 1);

                // save vertex id in vertex matrix
                linalgcu_matrix_set_element(vertices, (linalgcuSize_t)mesh->vertexCount, i, j);

                mesh->vertexCount++;
            }
            else {
                // invalidate vertex
                linalgcu_matrix_set_element(vertices, NAN, i, j);
            }
        }
    }

    // create boundary matrix
    error = linalgcu_matrix_create(&mesh->boundary,
        (linalgcuSize_t)(3.0 * M_PI * mesh->radius / mesh->distance), 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&vertices);
        fastect_mesh_release(&mesh);

        return error;
    }

    // get boundary vertices
    linalgcuMatrixData_t boundary[mesh->boundary->rows * 2];
    linalgcuMatrixData_t id;
    linalgcuMatrixData_t angle;

    for (linalgcuSize_t i = 0; i < mesh->vertexCount; i++) {
        // get vertex
        linalgcu_matrix_get_element(mesh->vertices, &x, i, 0);
        linalgcu_matrix_get_element(mesh->vertices, &y, i, 1);

        // calc radius and angle
        radius = sqrt(x * x + y * y);
        angle = fastect_mesh_angle(x, y);
        angle += angle < 0.0f ? 2.0 * M_PI : 0.0f;

        // check radius and angle
        if (radius >= mesh->radius - mesh->distance / 4.0f) {
            boundary[mesh->boundaryCount * 2 + 0] = (linalgcuMatrixData_t)i;
            boundary[mesh->boundaryCount * 2 + 1] = angle;
            mesh->boundaryCount++;
        }
    }

    // sort boundary
    qsort(boundary, mesh->boundaryCount, 2 * sizeof(linalgcuMatrixData_t), compare_boundary);

    // write boundary ids to matrix
    for (linalgcuSize_t i = 0; i < mesh->boundaryCount; i++) {
        linalgcu_matrix_set_element(mesh->boundary, boundary[i * 2], i, 0);
    }

    // create elements
    error = linalgcu_matrix_create(&mesh->elements,
        mesh->vertexCount * 6, 3, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&vertices);
        fastect_mesh_release(&mesh);

        return error;
    }

    linalgcuMatrixData_t neighbours[6];
    linalgcuMatrixData_t node;

    for (linalgcuSize_t i = 0; i < vertices->rows; i++) {
        for (linalgcuSize_t j = 0; j < vertices->columns; j++) {
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
                if (i < vertices->rows - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[2], i + 1, j);
                }
                else {
                    neighbours[2] = NAN;
                }

                // bottom right
                if (j < vertices-> columns - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[3], i, j + 1);
                }
                else {
                    neighbours[3] = NAN;
                }

                // bottom left
                if ((i > 0) && (j < vertices->columns - 1)) {
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
                if ((i < vertices->rows - 1) && (j > 0)) {
                    linalgcu_matrix_get_element(vertices, &neighbours[1], i + 1, j - 1);
                }
                else {
                    neighbours[1] = NAN;
                }

                // right
                if (i < vertices->rows - 1) {
                    linalgcu_matrix_get_element(vertices, &neighbours[2], i + 1, j);
                }
                else {
                    neighbours[2] = NAN;
                }

                // bottom right
                if ((i < vertices->rows - 1) && (j < vertices->columns - 1)) {
                    linalgcu_matrix_get_element(vertices, &neighbours[3], i + 1, j + 1);
                }
                else {
                    neighbours[3] = NAN;
                }

                // bottom left
                if (j < vertices->columns - 1) {
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
                    linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->elementCount, 1);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[2],
                        mesh->elementCount, 2);

                    mesh->elementCount++;
                }

                if (!isnan(neighbours[4])) {
                    linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[4],
                        mesh->elementCount, 1);
                    linalgcu_matrix_set_element(mesh->elements, neighbours[3],
                        mesh->elementCount, 2);

                    mesh->elementCount++;
                }
            }

            // create elements bottom left
            if ((isnan(neighbours[5])) && (!isnan(neighbours[4])) && (!isnan(neighbours[0]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[0], mesh->elementCount, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[4], mesh->elementCount, 2);
                mesh->elementCount++;
            }

            // create elements bottom left
            if ((isnan(neighbours[2])) && (!isnan(neighbours[1])) && (!isnan(neighbours[3]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[3], mesh->elementCount, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[1], mesh->elementCount, 2);
                mesh->elementCount++;
            }

            // create elements bottom left
            if ((isnan(neighbours[0])) && (!isnan(neighbours[1])) && (!isnan(neighbours[5]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[1], mesh->elementCount, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[5], mesh->elementCount, 2);
                mesh->elementCount++;
            }

            // create elements bottom left
            if ((isnan(neighbours[1])) && (!isnan(neighbours[2])) && (!isnan(neighbours[0]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[2], mesh->elementCount, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[0], mesh->elementCount, 2);
                mesh->elementCount++;
            }

            // create elements bottom left
            if ((isnan(neighbours[4])) && (!isnan(neighbours[5])) && (!isnan(neighbours[3]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[5], mesh->elementCount, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[3], mesh->elementCount, 2);
                mesh->elementCount++;
            }

            // create elements bottom left
            if ((isnan(neighbours[3])) && (!isnan(neighbours[4])) && (!isnan(neighbours[2]))) {
                linalgcu_matrix_set_element(mesh->elements, node, mesh->elementCount, 0);
                linalgcu_matrix_set_element(mesh->elements, neighbours[4], mesh->elementCount, 1);
                linalgcu_matrix_set_element(mesh->elements, neighbours[2], mesh->elementCount, 2);
                mesh->elementCount++;
            }
        }
    }

    // cleanup
    linalgcu_matrix_release(&vertices);

    // copy matrices to device
    linalgcu_matrix_copy_to_device(mesh->vertices, LINALGCU_TRUE, stream);
    linalgcu_matrix_copy_to_device(mesh->elements, LINALGCU_TRUE, stream);

    // set mesh pointer
    *meshPointer = mesh;

    return LINALGCU_SUCCESS;
}

linalgcuError_t fastect_mesh_release(fastectMesh_t* meshPointer) {
    // check input
    if ((meshPointer == NULL) || (*meshPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get mesh
    fastectMesh_t mesh = *meshPointer;

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
