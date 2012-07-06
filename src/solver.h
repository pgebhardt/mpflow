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

// solver struct
typedef struct {
    ert_grid_t grid;
    ert_gradient_solver_t gradient_solver;
    ert_electrodes_t electrodes;
    linalgcl_matrix_t voltage_calculation;
    linalgcl_matrix_t sigma;
    linalgcl_matrix_t current;
    linalgcl_matrix_t voltage;
    linalgcl_matrix_t f;
    linalgcl_matrix_t phi;
} ert_solver_s;
typedef ert_solver_s* ert_solver_t;

// create solver
linalgcl_error_t ert_solver_create(ert_solver_t* solverPointer,
    ert_mesh_t mesh, ert_electrodes_t electrodes,
    linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_device_id device_id, cl_command_queue queue);

// release solver
linalgcl_error_t ert_solver_release(ert_solver_t* solverPointer);

// forward solving
linalgcl_error_t ert_solver_forward_solve(ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue);

#endif
