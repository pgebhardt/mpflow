// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create conjugate solver
linalgcuError_t fastect_conjugate_solver_create(fastectConjugateSolver_t* solverPointer,
    linalgcuSize_t rows, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (rows <= 1) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create solver struct
    fastectConjugateSolver_t solver = malloc(sizeof(fastectConjugateSolver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->rows = rows;
    solver->residuum = NULL;
    solver->projection = NULL;
    solver->rsold = NULL;
    solver->rsnew = NULL;
    solver->tempVector = NULL;
    solver->tempNumber = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&solver->residuum, solver->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->projection, solver->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->rsold, solver->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->rsnew, solver->rows, 1, stream);
    error |= linalgcu_matrix_create(&solver->tempVector, solver->rows, solver->rows /
        LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&solver->tempNumber, solver->rows, 1, stream);

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
linalgcuError_t fastect_conjugate_solver_release(fastectConjugateSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectConjugateSolver_t solver = *solverPointer;

    // release matrices
    linalgcu_matrix_release(&solver->residuum);
    linalgcu_matrix_release(&solver->projection);
    linalgcu_matrix_release(&solver->rsold);
    linalgcu_matrix_release(&solver->rsnew);
    linalgcu_matrix_release(&solver->tempVector);
    linalgcu_matrix_release(&solver->tempNumber);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// solve conjugate
linalgcuError_t fastect_conjugate_solver_solve(fastectConjugateSolver_t solver,
    linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (A == NULL) || (x == NULL) || (f == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // temp for pointer swap
    linalgcuMatrix_t temp = NULL;

    // calc residuum r = f - A * x
    error  = linalgcu_matrix_multiply(solver->residuum, A, x, handle, stream);
    error |= linalgcu_matrix_scalar_multiply(solver->residuum, -1.0, stream);
    error |= linalgcu_matrix_add(solver->residuum, f, stream);

    // p = r
    error |= linalgcu_matrix_copy(solver->projection, solver->residuum, stream);

    // calc rsold
    error |= linalgcu_matrix_vector_dot_product(solver->rsold, solver->residuum, solver->residuum,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // iterate
    for (linalgcuSize_t i = 0; i < iterations; i++) {
        // calc A * p
        error  = fastect_conjugate_gemv(A, solver->projection, solver->tempVector, stream);

        // calc p * A * p
        error |= linalgcu_matrix_vector_dot_product(solver->tempNumber, solver->projection,
            solver->tempVector, stream);

        // update residuum
        error |= fastect_conjugate_update_vector(solver->residuum, solver->residuum, -1.0f,
            solver->tempVector, solver->rsold, solver->tempNumber, stream);

        // update x
        error |= fastect_conjugate_update_vector(x, x, 1.0f, solver->projection, solver->rsold,
            solver->tempNumber, stream);

        // calc rsnew
        error |= linalgcu_matrix_vector_dot_product(solver->rsnew, solver->residuum,
            solver->residuum, stream);

        // update projection
        error |= fastect_conjugate_update_vector(solver->projection, solver->residuum, 1.0f,
            solver->projection, solver->rsnew, solver->rsold, stream);

        // swap rsold and rsnew
        temp = solver->rsold;
        solver->rsold = solver->rsnew;
        solver->rsnew = temp;

        // check success
        if (error != LINALGCU_SUCCESS) {
            return error;
        }
    }

    return LINALGCU_SUCCESS;
}
