// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_CONJUGATE_H
#define FASTEIT_CONJUGATE_H

// conjugate solver struct
typedef struct {
    linalgcuSize_t rows;
    linalgcuMatrix_t residuum;
    linalgcuMatrix_t projection;
    linalgcuMatrix_t rsold;
    linalgcuMatrix_t rsnew;
    linalgcuMatrix_t tempVector;
    linalgcuMatrix_t tempNumber;
} fasteitConjugateSolver_s;
typedef fasteitConjugateSolver_s* fasteitConjugateSolver_t;

// create solver
linalgcuError_t fasteit_conjugate_solver_create(fasteitConjugateSolver_t* solverPointer,
    linalgcuSize_t rows, cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcuError_t fasteit_conjugate_solver_release(fasteitConjugateSolver_t* solverPointer);

// add scalar
LINALGCU_EXTERN_C
linalgcuError_t fasteit_conjugate_add_scalar(linalgcuMatrix_t vector,
    linalgcuMatrix_t scalar, linalgcuSize_t rows, linalgcuSize_t columns,
    cudaStream_t stream);

// update vector
LINALGCU_EXTERN_C
linalgcuError_t fasteit_conjugate_update_vector(linalgcuMatrix_t result,
    linalgcuMatrix_t x1, linalgcuMatrixData_t sign, linalgcuMatrix_t x2,
    linalgcuMatrix_t r1, linalgcuMatrix_t r2, cudaStream_t stream);

// fast gemv
LINALGCU_EXTERN_C
linalgcuError_t fasteit_conjugate_gemv(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
    linalgcuMatrix_t vector, cudaStream_t stream);

// solve conjugate
linalgcuError_t fasteit_conjugate_solver_solve(fasteitConjugateSolver_t self,
    linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, cublasHandle_t handle, cudaStream_t stream);

#endif
