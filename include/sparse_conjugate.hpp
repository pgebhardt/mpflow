// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SPARSE_CONJUGATE_H
#define FASTEIT_SPARSE_CONJUGATE_H

// conjugate solver struct
typedef struct {
    linalgcuSize_t rows;
    linalgcuSize_t columns;
    linalgcuMatrix_t residuum;
    linalgcuMatrix_t projection;
    linalgcuMatrix_t rsold;
    linalgcuMatrix_t rsnew;
    linalgcuMatrix_t tempVector;
    linalgcuMatrix_t tempNumber;
} fasteitSparseConjugateSolver_s;
typedef fasteitSparseConjugateSolver_s* fasteitSparseConjugateSolver_t;

// create solver
linalgcuError_t fasteit_sparse_conjugate_solver_create(
    fasteitSparseConjugateSolver_t* solverPointer, linalgcuSize_t rows,
    linalgcuSize_t columns, cudaStream_t stream);

// release solver
linalgcuError_t fasteit_sparse_conjugate_solver_release(
    fasteitSparseConjugateSolver_t* solverPointer);

// solve conjugate sparse
linalgcuError_t fasteit_sparse_conjugate_solver_solve(fasteitSparseConjugateSolver_t self,
    linalgcuSparseMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, linalgcuBool_t dcFree, cudaStream_t stream);

#endif
