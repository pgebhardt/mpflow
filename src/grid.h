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

#ifndef ERT_GRID_H
#define ERT_GRID_H

// solver program struct
typedef struct {
    cl_program program;
    cl_kernel kernel_update_system_matrix;
    cl_kernel kernel_unfold_system_matrix;
    cl_kernel kernel_regulize_system_matrix;
} ert_grid_program_s;
typedef ert_grid_program_s* ert_grid_program_t;

// solver grid struct
typedef struct {
    ert_mesh_t mesh;
    linalgcl_sparse_matrix_t system_matrix;
    linalgcl_matrix_t exitation_matrix;
    linalgcl_sparse_matrix_t gradient_matrix_sparse;
    linalgcl_matrix_t gradient_matrix;
    linalgcl_matrix_t sigma;
    linalgcl_matrix_t area;
    linalgcl_sparse_matrix_t restrict_phi;
    linalgcl_sparse_matrix_t restrict_sigma;
    linalgcl_sparse_matrix_t prolongate_phi;
    linalgcl_matrix_t x;
    linalgcl_matrix_t f;
    linalgcl_matrix_t r;
    linalgcl_matrix_t e;
} ert_grid_s;
typedef ert_grid_s* ert_grid_t;

// create new grid program
linalgcl_error_t ert_grid_program_create(ert_grid_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release grid program
linalgcl_error_t ert_grid_program_release(ert_grid_program_t* programPointer);

// create grid
linalgcl_error_t ert_grid_create(ert_grid_t* gridPointer,
    linalgcl_matrix_program_t matrix_program, ert_mesh_t mesh,
    cl_context context, cl_command_queue queue);

// release grid
linalgcl_error_t ert_grid_release(ert_grid_t* gridPointer);

// init system matrix
linalgcl_error_t ert_grid_init_system_matrix(ert_grid_t grid,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue);

// update system matrix
linalgcl_error_t ert_grid_update_system_matrix(ert_grid_t grid,
    ert_grid_program_t grid_program, cl_command_queue queue);

// init exitation matrix
linalgcl_error_t ert_grid_init_exitation_matrix(ert_grid_t grid,
    cl_context context, cl_command_queue queue);

// init intergrid transfer matrices
linalgcl_error_t ert_grid_init_intergrid_transfer_matrices(ert_grid_t grid,
    ert_grid_t finer_grid, ert_grid_t coarser_grid, 
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue);

#endif
