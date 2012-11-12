// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_MESH_H
#define FASTEIT_MESH_H

// mesh struct
typedef struct {
    linalgcuMatrixData_t radius;
    linalgcuMatrixData_t height;
    linalgcuSize_t nodeCount;
    linalgcuSize_t elementCount;
    linalgcuSize_t boundaryCount;
    linalgcuMatrix_t nodes;
    linalgcuMatrix_t elements;
    linalgcuMatrix_t boundary;
} fasteitMesh_s;
typedef fasteitMesh_s* fasteitMesh_t;

// create new mesh
linalgcuError_t fasteit_mesh_create(fasteitMesh_t* meshPointer,
    linalgcuMatrix_t nodes, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
    linalgcuSize_t nodeCount, linalgcuSize_t elementCount, linalgcuSize_t boundaryCount,
    linalgcuMatrixData_t radius, linalgcuMatrixData_t height);

// cleanup mesh
linalgcuError_t fasteit_mesh_release(fasteitMesh_t* meshPointer);

#endif
