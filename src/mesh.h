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

#ifndef ERT_MESH_H
#define ERT_MESH_H

#define ERT_MESH_MAX_VERTICES (40000)

// mesh struct
typedef struct {
    linalg_matrix_data_t radius;
    linalg_size_t vortex_count;
    linalg_matrix_t vertices;
} ert_mesh_s;
typedef ert_mesh_s* ert_mesh_t;

// create new mesh
linalg_error_t ert_mesh_create(ert_mesh_t* meshPointer,
    linalg_matrix_data_t radius, linalg_matrix_data_t distance);

// cleanup mesh
linalg_error_t ert_mesh_release(ert_mesh_t* meshPointer);

#endif
