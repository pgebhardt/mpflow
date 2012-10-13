// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_CONJUGATE_SPARSE_H
#define FASTEIT_CONJUGATE_SPARSE_H

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
} fasteitConjugateSparseSolver_s;
typedef fasteitConjugateSparseSolver_s* fasteitConjugateSparseSolver_t;

// create solver
linalgcuError_t fasteit_conjugate_sparse_solver_create(
    fasteitConjugateSparseSolver_t* solverPointer, linalgcuSize_t rows,
    linalgcuSize_t columns, cudaStream_t stream);

// release solver
linalgcuError_t fasteit_conjugate_sparse_solver_release(
    fasteitConjugateSparseSolver_t* solverPointer);

// add scalar
LINALGCU_EXTERN_C
linalgcuError_t fasteit_conjugate_sparse_add_scalar(linalgcuMatrix_t vector,
    linalgcuMatrix_t scalar, linalgcuSize_t rows, linalgcuSize_t columns,
    cudaStream_t stream);

// update vector
LINALGCU_EXTERN_C
linalgcuError_t fasteit_conjugate_sparse_update_vector(linalgcuMatrix_t result,
    linalgcuMatrix_t x1, linalgcuMatrixData_t sign, linalgcuMatrix_t x2,
    linalgcuMatrix_t r1, linalgcuMatrix_t r2, cudaStream_t stream);

// solve conjugate_sparse
linalgcuError_t fasteit_conjugate_sparse_solver_solve(fasteitConjugateSparseSolver_t self,
    linalgcuSparseMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, linalgcuBool_t dcFree, cudaStream_t stream);

#endif
