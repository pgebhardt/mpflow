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

#ifndef FASTECT_FORWARD_SOLVER_H
#define FASTECT_FORWARD_SOLVER_H

// solver struct
typedef struct {
    fastect_grid_t grid;
    fastect_conjugate_solver_t conjugate_solver;
    fastect_electrodes_t electrodes;
    linalgcu_size_t count;
    linalgcu_matrix_t phi;
    linalgcu_matrix_t* f;
    linalgcu_matrix_t voltage_calculation;
} fastect_forward_solver_s;
typedef fastect_forward_solver_s* fastect_forward_solver_t;

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t drive_pattern, linalgcu_matrix_t measurment_pattern,
    cublasHandle_t handle, cudaStream_t stream);

// release forward_solver
linalgcu_error_t fastect_forward_solver_release(fastect_forward_solver_t* solverPointer);

// calc excitaion
linalgcu_error_t fastect_forward_solver_calc_excitaion(fastect_forward_solver_t _solver,
    linalgcu_matrix_t drive_patternm, cublasHandle_t handle, cudaStream_t stream);

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream);

#endif
