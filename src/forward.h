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

#ifndef ERT_FORWARD_SOLVER_H
#define ERT_FORWARD_SOLVER_H

// solver program struct
typedef struct {
    cl_program program;
    cl_kernel kernel_copy_to_column;
    cl_kernel kernel_copy_from_column;
} ert_forward_solver_program_s;
typedef ert_forward_solver_program_s* ert_forward_solver_program_t;

// solver struct
typedef struct {
    ert_forward_solver_program_t program;
    ert_grid_t grid;
    ert_conjugate_solver_t conjugate_solver;
    ert_electrodes_t electrodes;
    linalgcl_size_t count;
    linalgcl_matrix_t pattern;
    linalgcl_matrix_t phi;
    linalgcl_matrix_t* f;
} ert_forward_solver_s;
typedef ert_forward_solver_s* ert_forward_solver_t;

// create new forward_solver program
linalgcl_error_t ert_forward_solver_program_create(ert_forward_solver_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release forward_solver program
linalgcl_error_t ert_forward_solver_program_release(ert_forward_solver_program_t* programPointer);

// create forward_solver
linalgcl_error_t ert_forward_solver_create(ert_forward_solver_t* solverPointer,
    ert_mesh_t mesh, ert_electrodes_t electrodes, linalgcl_size_t count,
    linalgcl_matrix_t pattern, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_device_id device, cl_command_queue queue);

// release forward_solver
linalgcl_error_t ert_forward_solver_release(ert_forward_solver_t* solverPointer);

// copy to column
linalgcl_error_t ert_forward_solver_copy_to_column(ert_forward_solver_program_t program,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue);

// copy from column
linalgcl_error_t ert_forward_solver_copy_from_column(ert_forward_solver_program_t program,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue);

// calc excitaion
linalgcl_error_t ert_forward_solver_calc_excitaion(ert_forward_solver_t _solver,
    linalgcl_matrix_program_t matrix_program, cl_context context, cl_command_queue queue);

// forward solving
actor_error_t ert_forward_solver_solve(actor_process_t self, ert_forward_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue);

#endif
