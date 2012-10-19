// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fasteit.h"

// create conjugate solver
linalgcuError_t fasteit_sparse_conjugate_solver_create(
    fasteitSparseConjugateSolver_t* solverPointer, linalgcuSize_t rows, linalgcuSize_t columns,
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
    fasteitSparseConjugateSolver_t self = malloc(sizeof(fasteitSparseConjugateSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->rows = rows;
    self->columns = columns;
    self->residuum = NULL;
    self->projection = NULL;
    self->rsold = NULL;
    self->rsnew = NULL;
    self->tempVector = NULL;
    self->tempNumber = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&self->residuum, self->rows, self->columns, stream);
    error |= linalgcu_matrix_create(&self->projection, self->rows, self->columns, stream);
    error |= linalgcu_matrix_create(&self->rsold, self->rows, self->columns, stream);
    error |= linalgcu_matrix_create(&self->rsnew, self->rows, self->columns, stream);
    error |= linalgcu_matrix_create(&self->tempVector, self->rows, self->columns, stream);
    error |= linalgcu_matrix_create(&self->tempNumber, self->rows, self->columns, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_sparse_conjugate_solver_release(&self);

        return error;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fasteit_sparse_conjugate_solver_release(
    fasteitSparseConjugateSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fasteitSparseConjugateSolver_t self = *solverPointer;

    // release matrices
    linalgcu_matrix_release(&self->residuum);
    linalgcu_matrix_release(&self->projection);
    linalgcu_matrix_release(&self->rsold);
    linalgcu_matrix_release(&self->rsnew);
    linalgcu_matrix_release(&self->tempVector);
    linalgcu_matrix_release(&self->tempNumber);

    // free struct
    free(self);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCU_SUCCESS;
}

// solve conjugate sparse
linalgcuError_t fasteit_sparse_conjugate_solver_solve(fasteitSparseConjugateSolver_t self,
    linalgcuSparseMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f, linalgcuSize_t iterations,
    linalgcuBool_t dcFree, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (A == NULL) || (x == NULL) || (f == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // temp for pointer swap
    linalgcuMatrix_t temp = NULL;

    // calc residuum r = f - A * x
    error  = linalgcu_sparse_matrix_multiply(self->residuum, A, x, stream);

    // regularize for dc free solution
    if (dcFree == LINALGCU_TRUE) {
        error |= linalgcu_matrix_sum(self->tempNumber, x, stream);
        error |= fasteit_conjugate_add_scalar(self->residuum, self->tempNumber,
            self->rows, self->columns, stream);
    }

    error |= linalgcu_matrix_scalar_multiply(self->residuum, -1.0, stream);
    error |= linalgcu_matrix_add(self->residuum, f, stream);

    // p = r
    error |= linalgcu_matrix_copy(self->projection, self->residuum, stream);

    // calc rsold
    error |= linalgcu_matrix_vector_dot_product(self->rsold, self->residuum,
        self->residuum, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // iterate
    for (linalgcuSize_t i = 0; i < iterations; i++) {
        // calc A * p
        error |= linalgcu_sparse_matrix_multiply(self->tempVector, A, self->projection,
            stream);

        // regularize for dc free solution
        if (dcFree == LINALGCU_TRUE) {
            error |= linalgcu_matrix_sum(self->tempNumber, self->projection, stream);
            error |= fasteit_conjugate_add_scalar(self->tempVector, self->tempNumber,
                self->rows, self->columns, stream);
        }

        // calc p * A * p
        error |= linalgcu_matrix_vector_dot_product(self->tempNumber, self->projection,
            self->tempVector, stream);

        // update residuum
        error |= fasteit_conjugate_update_vector(self->residuum, self->residuum,
            -1.0f, self->tempVector, self->rsold, self->tempNumber, stream);

        // update x
        error |= fasteit_conjugate_update_vector(x, x, 1.0f, self->projection,
            self->rsold, self->tempNumber, stream);

        // calc rsnew
        error |= linalgcu_matrix_vector_dot_product(self->rsnew, self->residuum,
            self->residuum, stream);

        // update projection
        error |= fasteit_conjugate_update_vector(self->projection, self->residuum,
            1.0f, self->projection, self->rsnew, self->rsold, stream);

        // swap rsold and rsnew
        temp = self->rsold;
        self->rsold = self->rsnew;
        self->rsnew = temp;

        // check success
        if (error != LINALGCU_SUCCESS) {
            return error;
        }
    }

    return LINALGCU_SUCCESS;
}
