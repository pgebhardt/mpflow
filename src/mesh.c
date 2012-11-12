// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fasteit.h"

linalgcuError_t fasteit_mesh_create(fasteitMesh_t* meshPointer,
    linalgcuMatrix_t nodes, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
    linalgcuSize_t nodeCount, linalgcuSize_t elementCount, linalgcuSize_t boundaryCount,
    linalgcuMatrixData_t radius, linalgcuMatrixData_t height, cudaStream_t stream) {
    // check input
    if ((meshPointer == NULL) || (nodes == NULL) || (elements == NULL) ||
        (boundary == NULL) || (nodeCount > nodes->rows) ||
        (elementCount > elements->rows) || (boundaryCount > boundary->rows) ||
        (radius <= 0.0f) || (height <= 0.0f)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init mesh pointer
    *meshPointer = NULL;

    // create mesg struct
    fasteitMesh_t self = malloc(sizeof(fasteitMesh_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->radius = radius;
    self->height = height;
    self->nodeCount = nodeCount;
    self->elementCount = elementCount;
    self->boundaryCount = boundaryCount;
    self->nodes = nodes;
    self->elements = elements;
    self->boundary = boundary;

    // set mesh pointer
    *meshPointer = self;

    return LINALGCU_SUCCESS;
}

linalgcuError_t fasteit_mesh_release(fasteitMesh_t* meshPointer) {
    // check input
    if ((meshPointer == NULL) || (*meshPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get mesh
    fasteitMesh_t self = *meshPointer;

    // cleanup nodes
    linalgcu_matrix_release(&self->nodes);

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
