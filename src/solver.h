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

#ifndef ERT_SOLVER_H
#define ERT_SOLVER_H

// solver program struct
typedef struct {
    cl_program program;
    cl_kernel kernel_update_system_matrix;
} ert_solver_program_s;
typedef ert_solver_program_s* ert_solver_program_t;

// solver grid struct
typedef struct {
    ert_mesh_t mesh;
    linalgcl_sparse_matrix_t system_matrix;
    linalgcl_matrix_t exitation_matrix;
    linalgcl_sparse_matrix_t gradient_matrix_sparse;
    linalgcl_matrix_t gradient_matrix;
    linalgcl_matrix_t sigma;
    linalgcl_matrix_t area;
} ert_solver_grid_s;
typedef ert_solver_grid_s* ert_solver_grid_t;

// solver struct
typedef struct {
    ert_solver_grid_t* grids;
    linalgcl_size_t grid_count;
    linalgcl_size_t max_grids;
    ert_solver_program_t program;
} ert_solver_s;
typedef ert_solver_s* ert_solver_t;

// create solver
linalgcl_error_t ert_solver_create(ert_solver_t* solverPointer,
    linalgcl_size_t max_grids, cl_context context, cl_device_id device_id);

// release solver
linalgcl_error_t ert_solver_release(ert_solver_t* solverPointer);

// add coarser grid
linalgcl_error_t ert_solver_add_coarser_grid(ert_solver_t solver,
    ert_mesh_t mesh, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_command_queue queue);

// create new solver program
linalgcl_error_t ert_solver_program_create(ert_solver_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release solver program
linalgcl_error_t ert_solver_program_release(ert_solver_program_t* programPointer);

// create solver grid
linalgcl_error_t ert_solver_grid_create(ert_solver_grid_t* gridPointer,
    ert_solver_t solver, linalgcl_matrix_program_t matrix_program,
    ert_mesh_t mesh, cl_context context, cl_command_queue queue);

// release solver grid
linalgcl_error_t ert_solver_grid_release(ert_solver_grid_t* gridPointer);

// update system matrix
linalgcl_error_t ert_solver_update_system_matrix(ert_solver_grid_t grid,
    ert_solver_t solver, cl_command_queue queue);

#endif
