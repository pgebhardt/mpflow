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

#ifndef ERT_MINRES_H
#define ERT_MINRES_H

// minres solver program struct
typedef struct {
    cl_program program;
} ert_minres_solver_program_s;
typedef ert_minres_solver_program_s* ert_minres_solver_program_t;

// minres solver struct
typedef struct {
    ert_grid_t grid;
    linalgcl_matrix_t residuum;
    linalgcl_matrix_t projection[3];
    linalgcl_matrix_t solution[3];
    ert_minres_solver_program_t program;
} ert_minres_solver_s;
typedef ert_minres_solver_s* ert_minres_solver_t;

// create new minres program
linalgcl_error_t ert_minres_solver_program_create(ert_minres_solver_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release minres program
linalgcl_error_t ert_minres_solver_program_release(ert_minres_solver_program_t* programPointer);

// create solver
linalgcl_error_t ert_minres_solver_create(ert_minres_solver_t* solverPointer,
    ert_grid_t grid, cl_context context, cl_device_id device_id, cl_command_queue queue);

// release solver
linalgcl_error_t ert_minres_solver_release(ert_minres_solver_t* solverPointer);

// solve minres
linalgcl_error_t ert_minres_solver_solve(ert_minres_solver_t solver,
    linalgcl_matrix_t x, linalgcl_matrix_t f,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue);

#endif
