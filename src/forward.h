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

// solver struct
typedef struct {
    ert_grid_t grid;
    ert_conjugate_solver_t conjugate_solver;
    ert_electrodes_t electrodes;
    linalgcu_size_t count;
    linalgcu_matrix_t pattern;
    linalgcu_matrix_t phi;
    linalgcu_matrix_t temp;
    linalgcu_matrix_t* f;
} ert_forward_solver_s;
typedef ert_forward_solver_s* ert_forward_solver_t;

// create forward_solver
linalgcu_error_t ert_forward_solver_create(ert_forward_solver_t* solverPointer,
    ert_mesh_t mesh, ert_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t pattern, cublasHandle_t handle, cudaStream_t stream);

// release forward_solver
linalgcu_error_t ert_forward_solver_release(ert_forward_solver_t* solverPointer);

// calc excitaion
linalgcu_error_t ert_forward_solver_calc_excitaion(ert_forward_solver_t _solver,
    cublasHandle_t handle, cudaStream_t stream);

// forward solving
linalgcu_error_t ert_forward_solver_solve(ert_forward_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream);

#endif
