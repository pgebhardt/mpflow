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

#ifndef FASTECT_GRID_H
#define FASTECT_GRID_H

// solver grid struct
typedef struct {
    fastect_mesh_t mesh;
    linalgcu_sparse_matrix_t system_matrix;
    linalgcu_matrix_t excitation_matrix;
    linalgcu_sparse_matrix_t gradient_matrix_sparse;
    linalgcu_sparse_matrix_t gradient_matrix_transposed_sparse;
    linalgcu_matrix_t gradient_matrix_transposed;
    linalgcu_matrix_t sigma;
    linalgcu_matrix_t area;
} fastect_grid_s;
typedef fastect_grid_s* fastect_grid_t;

// create grid
linalgcu_error_t fastect_grid_create(fastect_grid_t* gridPointer,
    fastect_mesh_t mesh, cublasHandle_t handle, cudaStream_t stream);

// release grid
linalgcu_error_t fastect_grid_release(fastect_grid_t* gridPointer);

// init system matrix
linalgcu_error_t fastect_grid_init_system_matrix(fastect_grid_t grid,
    cublasHandle_t handle, cudaStream_t stream);

// update system matrix
LINALGCU_EXTERN_C
linalgcu_error_t fastect_grid_update_system_matrix(fastect_grid_t grid,
    cudaStream_t stream);

// init exitation matrix
linalgcu_error_t fastect_grid_init_exitation_matrix(fastect_grid_t grid,
    fastect_electrodes_t electrodes, cudaStream_t stream);

#endif
