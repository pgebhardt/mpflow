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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include "basis.h"
#include "mesh.h"
#include "conjugate.h"

static void print_matrix(linalgcu_matrix_t matrix) {
    if (matrix == NULL) {
        return;
    }

    // value memory
    linalgcu_matrix_data_t value = 0.0;

    for (linalgcu_size_t i = 0; i < matrix->size_m; i++) {
        for (linalgcu_size_t j = 0; j < matrix->size_n; j++) {
            // get value
            linalgcu_matrix_get_element(matrix, &value, i, j);

            printf("%f, ", value);
        }
        printf("\n");
    }
}

// create conjugate solver
linalgcu_error_t ert_conjugate_solver_create(ert_conjugate_solver_t* solverPointer,
    linalgcu_sparse_matrix_t system_matrix, linalgcu_size_t size,
    cublasHandle_t handle) {
    // check input
    if ((solverPointer == NULL) || (system_matrix == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create solver struct
    ert_conjugate_solver_t solver = malloc(sizeof(ert_conjugate_solver_s));

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
    error  = linalgcu_matrix_create(&solver->residuum, solver->size, 1);
    error |= linalgcu_matrix_create(&solver->projection, solver->size, 1);
    error |= linalgcu_matrix_create(&solver->rsold, solver->size, 1);
    error |= linalgcu_matrix_create(&solver->rsnew, solver->size, 1);
    error |= linalgcu_matrix_create(&solver->ones, solver->size, 1);
    error |= linalgcu_matrix_create(&solver->temp_matrix, solver->size, solver->size);
    error |= linalgcu_matrix_create(&solver->temp_vector, solver->size, 1);
    error |= linalgcu_matrix_create(&solver->temp_number, solver->size, 1);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        ert_conjugate_solver_release(&solver);

        return error;
    }

    // set ones
    for (linalgcu_size_t i = 0; i < solver->size; i++) {
        linalgcu_matrix_set_element(solver->ones, 1.0, i, 0);
    }
    error |= linalgcu_matrix_copy_to_device(solver->ones, LINALGCU_TRUE);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        ert_conjugate_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcu_error_t ert_conjugate_solver_release(ert_conjugate_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    ert_conjugate_solver_t solver = *solverPointer;

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
linalgcu_error_t ert_conjugate_solver_solve(ert_conjugate_solver_t solver,
    linalgcu_matrix_t x, linalgcu_matrix_t f, linalgcu_size_t iterations,
    cublasHandle_t handle) {
    // check input
    if ((solver == NULL) || (x == NULL) || (f == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // init matrices
    // calc residuum r = f - A * x
    linalgcu_matrix_vector_dot_product(solver->temp_number, x, solver->ones);
    linalgcu_sparse_matrix_vector_multiply(solver->residuum, solver->system_matrix, x);
    ert_conjugate_add_scalar(solver->residuum, solver->temp_number);
    linalgcu_matrix_scalar_multiply(solver->residuum, -1.0, handle);
    linalgcu_matrix_add(solver->residuum, f, handle);

    // p = r
    linalgcu_matrix_copy(solver->projection, solver->residuum, LINALGCU_TRUE);

    // calc rsold
    linalgcu_matrix_vector_dot_product(solver->rsold, solver->residuum, solver->residuum);

    // iterate
    for (linalgcu_size_t i = 0; i < iterations; i++) {
        /*if (solver->rsold / solver->size < 1E-8) {
            break;
        }*/

        // calc A * p
        linalgcu_matrix_vector_dot_product(solver->temp_number, solver->projection, solver->ones);
        linalgcu_sparse_matrix_vector_multiply(solver->temp_vector, solver->system_matrix,
            solver->projection);
        ert_conjugate_add_scalar(solver->temp_vector, solver->temp_number);

        // calc p * A * p
        linalgcu_matrix_vector_dot_product(solver->temp_number, solver->projection,
            solver->temp_vector);

        // update residuum
        ert_conjugate_udate_vector(solver->residuum, solver->residuum, -1.0f,
            solver->temp_vector, solver->rsold, solver->temp_number);

        // update x
        ert_conjugate_udate_vector(x, x, 1.0f, solver->projection, solver->rsold,
            solver->temp_number);

        // calc rsnew
        linalgcu_matrix_vector_dot_product(solver->rsnew, solver->residuum,
            solver->residuum);

        // update projection
        ert_conjugate_udate_vector(solver->projection, solver->residuum, 1.0f,
            solver->projection, solver->rsnew, solver->rsold);

        // update rsold
        linalgcu_matrix_t temp = solver->rsold;
        solver->rsold = solver->rsnew;
        solver->rsnew = temp;
    }

    return LINALGCU_SUCCESS;
}
