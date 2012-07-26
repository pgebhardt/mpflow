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

#ifndef FASTECT_MESH_H
#define FASTECT_MESH_H

// mesh struct
typedef struct {
    linalgcu_matrix_data_t radius;
    linalgcu_matrix_data_t distance;
    linalgcu_size_t vertex_count;
    linalgcu_size_t element_count;
    linalgcu_size_t boundary_count;
    linalgcu_matrix_t vertices;
    linalgcu_matrix_t elements;
    linalgcu_matrix_t boundary;
} fastect_mesh_s;
typedef fastect_mesh_s* fastect_mesh_t;

// create new mesh
linalgcu_error_t fastect_mesh_create(fastect_mesh_t* meshPointer,
    linalgcu_matrix_data_t radius, linalgcu_matrix_data_t distance,
    cudaStream_t stream);

// cleanup mesh
linalgcu_error_t fastect_mesh_release(fastect_mesh_t* meshPointer);

#endif
