// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_MESH_H
#define FASTECT_MESH_H

// mesh struct
typedef struct {
    linalgcuMatrixData_t radius;
    linalgcuMatrixData_t height;
    linalgcuSize_t vertexCount;
    linalgcuSize_t elementCount;
    linalgcuSize_t boundaryCount;
    linalgcuMatrix_t vertices;
    linalgcuMatrix_t elements;
    linalgcuMatrix_t boundary;
} fastectMesh_s;
typedef fastectMesh_s* fastectMesh_t;

// create new mesh
linalgcuError_t fastect_mesh_create(fastectMesh_t* meshPointer,
    linalgcuMatrix_t vertices, linalgcuMatrix_t elements, linalgcuMatrix_t boundary,
    linalgcuSize_t vertexCount, linalgcuSize_t elementCount, linalgcuSize_t boundaryCount,
    linalgcuMatrixData_t radius, linalgcuMatrixData_t height);

// cleanup mesh
linalgcuError_t fastect_mesh_release(fastectMesh_t* meshPointer);

#endif
