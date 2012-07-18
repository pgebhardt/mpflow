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

#ifndef ERT_FORWARD_H
#define ERT_FORWARD_H

// forward solver program struct
typedef struct {
    cl_program program;
    cl_kernel kernel_add_scalar;
    cl_kernel kernel_update_vector;
} ert_forward_solver_program_s;
typedef ert_forward_solver_program_s* ert_forward_solver_program_t;

// forward solver struct
typedef struct {
    linalgcl_size_t size;
    linalgcl_sparse_matrix_t system_matrix;
    linalgcl_matrix_t residuum;
    linalgcl_matrix_t projection;
    linalgcl_matrix_t rsold;
    linalgcl_matrix_t rsnew;
    linalgcl_matrix_t ones;
    linalgcl_matrix_t temp_matrix;
    linalgcl_matrix_t temp_vector;
    linalgcl_matrix_t temp_number;
    ert_forward_solver_program_t program;
} ert_forward_solver_s;
typedef ert_forward_solver_s* ert_forward_solver_t;

// create new forward program
linalgcl_error_t ert_forward_solver_program_create(ert_forward_solver_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release forward program
linalgcl_error_t ert_forward_solver_program_release(ert_forward_solver_program_t* programPointer);

// create solver
linalgcl_error_t ert_forward_solver_create(ert_forward_solver_t* solverPointer,
    linalgcl_sparse_matrix_t system_matrix, linalgcl_size_t size,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_device_id device_id, cl_command_queue queue);

// release solver
linalgcl_error_t ert_forward_solver_release(ert_forward_solver_t* solverPointer);

// update vector
linalgcl_error_t ert_forward_update_vector(ert_forward_solver_t solver,
    linalgcl_matrix_t result, linalgcl_matrix_t x1, linalgcl_matrix_data_t sign,
    linalgcl_matrix_t x2, linalgcl_matrix_t r1, linalgcl_matrix_t r2, cl_command_queue queue);

// solve forward
linalgcl_error_t ert_forward_solver_solve(ert_forward_solver_t solver,
    linalgcl_matrix_t x, linalgcl_matrix_t f, linalgcl_matrix_data_t tolerance,
    linalgcl_size_t max_iter,linalgcl_matrix_program_t matrix_program,
    cl_command_queue queue);

#endif
