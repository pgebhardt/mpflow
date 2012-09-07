// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_MESH_H
#define FASTECT_MESH_H

// mesh struct
typedef struct {
    linalgcu_matrix_data_t radius;
    linalgcu_matrix_data_t distance;
    linalgcu_matrix_data_t height;
    linalgcu_size_t vertexCount;
    linalgcu_size_t elementCount;
    linalgcu_size_t boundaryCount;
    linalgcu_matrix_t vertices;
    linalgcu_matrix_t elements;
    linalgcu_matrix_t boundary;
} fastect_mesh_s;
typedef fastect_mesh_s* fastect_mesh_t;

// create new mesh
linalgcu_error_t fastect_mesh_create(fastect_mesh_t* meshPointer,
    linalgcu_matrix_data_t radius, linalgcu_matrix_data_t distance,
    linalgcu_matrix_data_t height, cudaStream_t stream);

// cleanup mesh
linalgcu_error_t fastect_mesh_release(fastect_mesh_t* meshPointer);

#endif
