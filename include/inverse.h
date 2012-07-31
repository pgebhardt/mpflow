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

#ifndef FASTECT_INVERSE_SOLVER_H
#define FASTECT_INVERSE_SOLVER_H

// solver struct
typedef struct {
    fastect_conjugate_solver_t conjugate_solver;
    linalgcu_matrix_t jacobian;
    linalgcu_matrix_t dU;
    linalgcu_matrix_t dSigma;
    linalgcu_matrix_t f;
    linalgcu_matrix_t A;
    linalgcu_matrix_t regularization;
    linalgcu_matrix_data_t lambda;
} fastect_inverse_solver_s;
typedef fastect_inverse_solver_s* fastect_inverse_solver_t;

// create inverse_solver
linalgcu_error_t fastect_inverse_solver_create(fastect_inverse_solver_t* solverPointer,
    linalgcu_matrix_t jacobian, linalgcu_matrix_data_t lambda,
    cublasHandle_t handle, cudaStream_t stream);

// release inverse_solver
linalgcu_error_t fastect_inverse_solver_release(fastect_inverse_solver_t* solverPointer);

// calc system matrix
linalgcu_error_t fastect_inverse_solver_calc_system_matrix(fastect_inverse_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream);

// calc excitation
linalgcu_error_t fastect_inverse_solver_calc_excitation(fastect_inverse_solver_t solver,
    linalgcu_matrix_t calculated_voltage, linalgcu_matrix_t measured_voltage,
    cublasHandle_t handle, cudaStream_t stream);

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve(fastect_inverse_solver_t solver,
    linalgcu_matrix_t calculated_voltage, linalgcu_matrix_t measured_voltage,
    cublasHandle_t handle, cudaStream_t stream);

// linear inverse solving
linalgcu_error_t fastect_inverse_solver_solve_linear(fastect_inverse_solver_t solver,
    linalgcu_matrix_t calculated_voltage, linalgcu_matrix_t measured_voltage,
    cublasHandle_t handle, cudaStream_t stream);

#endif
