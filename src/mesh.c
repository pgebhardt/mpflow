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
        (boundary == NULL) || (vertexCount > vertices->rows) ||
        (elementCount > elements->rows) || (boundaryCount > boundary->rows) ||
        (radius <= 0.0f) || (height <= 0.0f)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init mesh pointer
    *meshPointer = NULL;

    // create mesg struct
    fastectMesh_t self = malloc(sizeof(fastectMesh_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->radius = radius;
    self->height = height;
    self->vertexCount = vertexCount;
    self->elementCount = elementCount;
    self->boundaryCount = boundaryCount;
    self->vertices = vertices;
    self->elements = elements;
    self->boundary = boundary;

    // set mesh pointer
    *meshPointer = self;

    return LINALGCU_SUCCESS;
}

linalgcuError_t fastect_mesh_release(fastectMesh_t* meshPointer) {
    // check input
    if ((meshPointer == NULL) || (*meshPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get mesh
    fastectMesh_t self = *meshPointer;

    // cleanup vertices
    linalgcu_matrix_release(&self->vertices);

    // cleanup elements
    linalgcu_matrix_release(&self->elements);

    // cleanup boundary
    linalgcu_matrix_release(&self->boundary);

    // free struct
    free(self);

    // set mesh pointer to NULL
    *meshPointer = NULL;

    return LINALGCU_SUCCESS;
}
