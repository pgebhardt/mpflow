// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

linalgcuError_t fastect_mesh_create(fastectMesh_t* meshPointer,
    linalgcuMatrix_t vertices, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
    linalgcuSize_t vertexCount, linalgcuSize_t elementCount, linalgcuSize_t boundaryCount,
    linalgcuMatrixData_t radius, linalgcuMatrixData_t height) {
    // check input
    if ((meshPointer == NULL) || (vertices == NULL) || (elements == NULL) ||
        (boundary == NULL) || (radius <= 0.0f) || (height <= 0.0f)) {
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
    mesh->height = height;
    mesh->vertexCount = vertexCount;
    mesh->elementCount = elementCount;
    mesh->boundaryCount = boundaryCount;
    mesh->vertices = vertices;
    mesh->elements = elements;
    mesh->boundary = boundary;

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
