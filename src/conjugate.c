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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include "basis.h"
#include "mesh.h"
#include "conjugate.h"

// create conjugate solver
linalgcu_error_t fastect_conjugate_solver_create(fastect_conjugate_solver_t* solverPointer,
    linalgcu_sparse_matrix_t system_matrix, linalgcu_size_t size,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (system_matrix == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create solver struct
    fastect_conjugate_solver_t solver = malloc(sizeof(fastect_conjugate_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->size = size;
    solver->system_matrix = NULL;
    solver->residuum = NULL;
    solver->projection = NULL;
    solver->rsold = NULL;
    solver->rsnew = NULL;
    solver->ones = NULL;
    solver->temp_matrix = NULL;
    solver->temp_vector = NULL;

    // set system_matrix
    solver->system_matrix = system_matrix;

    // create matrices
    error  = linalgcu_matrix_create(&solver->residuum, solver->size, 1, stream);
    error |= linalgcu_matrix_create(&solver->projection, solver->size, 1, stream);
    error |= linalgcu_matrix_create(&solver->rsold, solver->size, 1, stream);
    error |= linalgcu_matrix_create(&solver->rsnew, solver->size, 1, stream);
    error |= linalgcu_matrix_create(&solver->ones, solver->size, 1, stream);
    error |= linalgcu_matrix_create(&solver->temp_matrix, solver->size, solver->size, stream);
    error |= linalgcu_matrix_create(&solver->temp_vector, solver->size, 1, stream);
    error |= linalgcu_matrix_create(&solver->temp_number, solver->size, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_conjugate_solver_release(&solver);

        return error;
    }

    // set ones
    for (linalgcu_size_t i = 0; i < solver->size; i++) {
        linalgcu_matrix_set_element(solver->ones, 1.0, i, 0);
    }
    error |= linalgcu_matrix_copy_to_device(solver->ones, LINALGCU_TRUE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_conjugate_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t fastect_conjugate_solver_release(fastect_conjugate_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastect_conjugate_solver_t solver = *solverPointer;

    // release matrices
    linalgcu_matrix_release(&solver->residuum);
    linalgcu_matrix_release(&solver->projection);
    linalgcu_matrix_release(&solver->rsold);
    linalgcu_matrix_release(&solver->rsnew);
    linalgcu_matrix_release(&solver->ones);
    linalgcu_matrix_release(&solver->temp_matrix);
    linalgcu_matrix_release(&solver->temp_vector);
    linalgcu_matrix_release(&solver->temp_number);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// solve conjugate
linalgcu_error_t fastect_conjugate_solver_solve(fastect_conjugate_solver_t solver,
    linalgcu_matrix_t x, linalgcu_matrix_t f, linalgcu_size_t iterations,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (x == NULL) || (f == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // init matrices
    // calc residuum r = f - A * x
    linalgcu_matrix_vector_dot_product(solver->temp_number, x, solver->ones, stream);
    linalgcu_sparse_matrix_vector_multiply(solver->residuum, solver->system_matrix, x, stream);
    fastect_conjugate_add_scalar(solver->residuum, solver->temp_number, solver->size, stream);
    linalgcu_matrix_scalar_multiply(solver->residuum, -1.0, handle, stream);
    linalgcu_matrix_add(solver->residuum, f, handle, stream);

    // p = r
    linalgcu_matrix_copy(solver->projection, solver->residuum, LINALGCU_TRUE, stream);

    // calc rsold
    linalgcu_matrix_vector_dot_product(solver->rsold, solver->residuum, solver->residuum, stream);

    // iterate
    for (linalgcu_size_t i = 0; i < iterations; i++) {
        // calc A * p
        linalgcu_matrix_vector_dot_product(solver->temp_number, solver->projection, solver->ones, stream);
        linalgcu_sparse_matrix_vector_multiply(solver->temp_vector, solver->system_matrix,
            solver->projection, stream);
        fastect_conjugate_add_scalar(solver->temp_vector, solver->temp_number, solver->size, stream);

        // calc p * A * p
        linalgcu_matrix_vector_dot_product(solver->temp_number, solver->projection,
            solver->temp_vector, stream);

        // update residuum
        fastect_conjugate_udate_vector(solver->residuum, solver->residuum, -1.0f,
            solver->temp_vector, solver->rsold, solver->temp_number, stream);

        // update x
        fastect_conjugate_udate_vector(x, x, 1.0f, solver->projection, solver->rsold,
            solver->temp_number, stream);

        // calc rsnew
        linalgcu_matrix_vector_dot_product(solver->rsnew, solver->residuum,
            solver->residuum, stream);

        // update projection
        fastect_conjugate_udate_vector(solver->projection, solver->residuum, 1.0f,
            solver->projection, solver->rsnew, solver->rsold, stream);

        // update rsold
        linalgcu_matrix_t temp = solver->rsold;
        solver->rsold = solver->rsnew;
        solver->rsnew = temp;
    }

    return LINALGCU_SUCCESS;
}
