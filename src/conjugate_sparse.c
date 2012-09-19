// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create conjugate solver
linalgcuError_t fastect_conjugate_sparse_solver_create(
    fastectConjugateSparseSolver_t* solverPointer, linalgcuSize_t rows, linalgcuSize_t columns,
    cudaStream_t stream) {
    // check input
    if ((solverPointer == NULL) || (rows <= 1)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create solver struct
    fastectConjugateSparseSolver_t solver = malloc(sizeof(fastectConjugateSparseSolver_s));

    // check success
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    solver->rows = rows;
    solver->columns = columns;
    solver->residuum = NULL;
    solver->projection = NULL;
    solver->rsold = NULL;
    solver->rsnew = NULL;
    solver->tempVector = NULL;
    solver->tempNumber = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&solver->residuum, solver->rows, solver->columns, stream);
    error |= linalgcu_matrix_create(&solver->projection, solver->rows, solver->columns, stream);
    error |= linalgcu_matrix_create(&solver->rsold, solver->rows, solver->columns, stream);
    error |= linalgcu_matrix_create(&solver->rsnew, solver->rows, solver->columns, stream);
    error |= linalgcu_matrix_create(&solver->tempVector, solver->rows, solver->columns, stream);
    error |= linalgcu_matrix_create(&solver->tempNumber, solver->rows, solver->columns, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_conjugate_sparse_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fastect_conjugate_sparse_solver_release(fastectConjugateSparseSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectConjugateSparseSolver_t solver = *solverPointer;

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

// solve conjugate_sparse sparse
linalgcuError_t fastect_conjugate_sparse_solver_solve(fastectConjugateSparseSolver_t solver,
    linalgcuSparseMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f, linalgcuSize_t iterations,
    linalgcuBool_t dcFree, cudaStream_t stream) {
    // check input
    if ((solver == NULL) || (A == NULL) || (x == NULL) || (f == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // temp for pointer swap
    linalgcuMatrix_t temp = NULL;

    // calc residuum r = f - A * x
    error  = linalgcu_sparse_matrix_multiply(solver->residuum, A, x, stream);

    // regularize for dc free solution
    if (dcFree == LINALGCU_TRUE) {
        error |= linalgcu_matrix_sum(solver->tempNumber, x, stream);
        error |= fastect_conjugate_sparse_add_scalar(solver->residuum, solver->tempNumber,
            solver->rows, solver->columns, stream);
    }

    error |= linalgcu_matrix_scalar_multiply(solver->residuum, -1.0, stream);
    error |= linalgcu_matrix_add(solver->residuum, f, stream);

    // p = r
    error |= linalgcu_matrix_copy(solver->projection, solver->residuum, stream);

    // calc rsold
    error |= linalgcu_matrix_vector_dot_product(solver->rsold, solver->residuum,
        solver->residuum, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // iterate
    for (linalgcuSize_t i = 0; i < iterations; i++) {
        // calc A * p
        error |= linalgcu_sparse_matrix_multiply(solver->tempVector, A, solver->projection,
            stream);

        // regularize for dc free solution
        if (dcFree == LINALGCU_TRUE) {
            error |= linalgcu_matrix_sum(solver->tempNumber, solver->projection, stream);
            error |= fastect_conjugate_sparse_add_scalar(solver->tempVector, solver->tempNumber,
                solver->rows, solver->columns, stream);
        }

        // calc p * A * p
        error |= linalgcu_matrix_vector_dot_product(solver->tempNumber, solver->projection,
            solver->tempVector, stream);

        // update residuum
        error |= fastect_conjugate_sparse_update_vector(solver->residuum, solver->residuum,
            -1.0f, solver->tempVector, solver->rsold, solver->tempNumber, stream);

        // update x
        error |= fastect_conjugate_sparse_update_vector(x, x, 1.0f, solver->projection,
            solver->rsold, solver->tempNumber, stream);

        // calc rsnew
        error |= linalgcu_matrix_vector_dot_product(solver->rsnew, solver->residuum,
            solver->residuum, stream);

        // update projection
        error |= fastect_conjugate_sparse_update_vector(solver->projection, solver->residuum,
            1.0f, solver->projection, solver->rsnew, solver->rsold, stream);

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
