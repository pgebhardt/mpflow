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
    cl_kernel kernel_copy_to_column;
    cl_kernel kernel_copy_from_column;
    cl_kernel kernel_calc_jacobian;
    cl_kernel kernel_regularize_jacobian;
    cl_kernel kernel_calc_sigma_excitation;
} ert_solver_program_s;
typedef ert_solver_program_s* ert_solver_program_t;

// solver struct
typedef struct {
    ert_solver_program_t program;
    ert_grid_t grid;
    ert_forward_solver_t forward_solver;
    ert_electrodes_t electrodes;
    linalgcl_size_t measurment_count;
    linalgcl_size_t drive_count;
    linalgcl_matrix_t jacobian;
    linalgcl_matrix_t regularized_jacobian;
    linalgcl_matrix_t voltage_calculation;
    linalgcl_matrix_t sigma;
    linalgcl_matrix_t phi;
    linalgcl_matrix_t applied_phi;
    linalgcl_matrix_t lead_phi;
    linalgcl_matrix_t* applied_f;
    linalgcl_matrix_t* lead_f;
    linalgcl_matrix_t calculated_voltage;
    linalgcl_matrix_t measured_voltage;
} ert_solver_s;
typedef ert_solver_s* ert_solver_t;

// create new solver program
linalgcl_error_t ert_solver_program_create(ert_solver_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release solver program
linalgcl_error_t ert_solver_program_release(ert_solver_program_t* programPointer);

// create solver
linalgcl_error_t ert_solver_create(ert_solver_t* solverPointer,
    ert_mesh_t mesh, ert_electrodes_t electrodes, linalgcl_size_t measurment_count,
    linalgcl_size_t drive_count, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_device_id device_id, cl_command_queue queue);

// release solver
linalgcl_error_t ert_solver_release(ert_solver_t* solverPointer);

// copy to column
linalgcl_error_t ert_solver_copy_to_column(ert_solver_t solver,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue);

// copy from column
linalgcl_error_t ert_solver_copy_from_column(ert_solver_t solver,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue);

// calc excitaion
linalgcl_error_t ert_solver_calc_excitaion(ert_solver_t solver,
    linalgcl_matrix_t drive_pattern, linalgcl_matrix_t measurment_pattern,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue);

// calc jacobian
linalgcl_error_t ert_solver_calc_jacobian(ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue);

// forward solving
linalgcl_error_t ert_solver_forward_solve(ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue);

// inverse solving
linalgcl_error_t ert_solver_inverse_solve(ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue);

#endif
