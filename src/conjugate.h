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

#ifndef FASTECT_CONJUGATE_H
#define FASTECT_CONJUGATE_H

// conjugate solver struct
typedef struct {
    linalgcu_size_t size;
    linalgcu_sparse_matrix_t system_matrix;
    linalgcu_matrix_t residuum;
    linalgcu_matrix_t projection;
    linalgcu_matrix_t rsold;
    linalgcu_matrix_t rsnew;
    linalgcu_matrix_t temp_vector;
    linalgcu_matrix_t temp_number;
} fastect_conjugate_solver_s;
typedef fastect_conjugate_solver_s* fastect_conjugate_solver_t;

// create solver
linalgcu_error_t fastect_conjugate_solver_create(fastect_conjugate_solver_t* solverPointer,
    linalgcu_sparse_matrix_t system_matrix, linalgcu_size_t size,
    cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcu_error_t fastect_conjugate_solver_release(fastect_conjugate_solver_t* solverPointer);

// add scalar
LINALGCU_EXTERN_C
linalgcu_error_t fastect_conjugate_add_scalar(linalgcu_matrix_t vector, linalgcu_matrix_t scalar,
    linalgcu_size_t size, cudaStream_t stream);

// update vector
LINALGCU_EXTERN_C
linalgcu_error_t fastect_conjugate_udate_vector(linalgcu_matrix_t result,
    linalgcu_matrix_t x1, linalgcu_matrix_data_t sign, linalgcu_matrix_t x2,
    linalgcu_matrix_t r1, linalgcu_matrix_t r2, cudaStream_t stream);

// solve conjugate
linalgcu_error_t fastect_conjugate_solver_solve(fastect_conjugate_solver_t solver,
    linalgcu_matrix_t x, linalgcu_matrix_t f,
    linalgcu_size_t iterations, cublasHandle_t handle, cudaStream_t stream);

#endif
