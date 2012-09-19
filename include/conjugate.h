// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_CONJUGATE_H
#define FASTECT_CONJUGATE_H

// conjugate solver struct
typedef struct {
    linalgcuSize_t rows;
    linalgcuMatrix_t residuum;
    linalgcuMatrix_t projection;
    linalgcuMatrix_t rsold;
    linalgcuMatrix_t rsnew;
    linalgcuMatrix_t tempVector;
    linalgcuMatrix_t tempNumber;
} fastectConjugateSolver_s;
typedef fastectConjugateSolver_s* fastectConjugateSolver_t;

// create solver
linalgcuError_t fastect_conjugate_solver_create(fastectConjugateSolver_t* solverPointer,
    linalgcuSize_t rows, cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcuError_t fastect_conjugate_solver_release(fastectConjugateSolver_t* solverPointer);

// update vector
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_update_vector(linalgcuMatrix_t result,
    linalgcuMatrix_t x1, linalgcuMatrixData_t sign, linalgcuMatrix_t x2,
    linalgcuMatrix_t r1, linalgcuMatrix_t r2, cudaStream_t stream);

// fast gemv
LINALGCU_EXTERN_C
linalgcuError_t fastect_conjugate_gemv(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
    linalgcuMatrix_t vector, cudaStream_t stream);

// solve conjugate
linalgcuError_t fastect_conjugate_solver_solve(fastectConjugateSolver_t solver,
    linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, cublasHandle_t handle, cudaStream_t stream);

#endif
