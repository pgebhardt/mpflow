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
    fastectConjugateSolver_t self = malloc(sizeof(fastectConjugateSolver_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->rows = rows;
    self->residuum = NULL;
    self->projection = NULL;
    self->rsold = NULL;
    self->rsnew = NULL;
    self->tempVector = NULL;
    self->tempNumber = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&self->residuum, self->rows, 1, stream);
    error |= linalgcu_matrix_create(&self->projection, self->rows, 1, stream);
    error |= linalgcu_matrix_create(&self->rsold, self->rows, 1, stream);
    error |= linalgcu_matrix_create(&self->rsnew, self->rows, 1, stream);
    error |= linalgcu_matrix_create(&self->tempVector, self->rows, self->residuum->rows /
        LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&self->tempNumber, self->rows, 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fastect_conjugate_solver_release(&self);

        return error;
    }

    // set solver pointer
    *solverPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver
linalgcuError_t fastect_conjugate_solver_release(fastectConjugateSolver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get solver
    fastectConjugateSolver_t self = *solverPointer;

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

// solve conjugate
linalgcuError_t fastect_conjugate_solver_solve(fastectConjugateSolver_t self,
    linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (A == NULL) || (x == NULL) || (f == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // temp for pointer swap
    linalgcuMatrix_t temp = NULL;

    // calc residuum r = f - A * x
    error  = linalgcu_matrix_multiply(self->residuum, A, x, handle, stream);
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
        error  = fastect_conjugate_gemv(self->tempVector, A, self->projection, stream);

        // calc p * A * p
        error |= linalgcu_matrix_vector_dot_product(self->tempNumber, self->projection,
            self->tempVector, stream);

        // update residuum
        error |= fastect_conjugate_update_vector(self->residuum, self->residuum, -1.0f,
            self->tempVector, self->rsold, self->tempNumber, stream);

        // update x
        error |= fastect_conjugate_update_vector(x, x, 1.0f, self->projection, self->rsold,
            self->tempNumber, stream);

        // calc rsnew
        error |= linalgcu_matrix_vector_dot_product(self->rsnew, self->residuum,
            self->residuum, stream);

        // update projection
        error |= fastect_conjugate_update_vector(self->projection, self->residuum, 1.0f,
            self->projection, self->rsnew, self->rsold, stream);

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
