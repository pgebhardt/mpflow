// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_INVERSE_SOLVER_H
#define FASTECT_INVERSE_SOLVER_H

// solver struct
typedef struct {
    fastectConjugateSolver_t conjugate_solver;
    linalgcuMatrix_t deltaVoltage;
    linalgcuMatrix_t zeros;
    linalgcuMatrix_t excitation;
    linalgcuMatrix_t systemMatrix;
} fastectInverseSolver_s;
typedef fastectInverseSolver_s* fastectInverseSolver_t;

// create inverse_solver
linalgcuError_t fastect_inverse_solver_create(fastectInverseSolver_t* solverPointer,
    linalgcuSize_t elementCount, linalgcuSize_t voltageCount, linalgcuMatrix_t systemMatrix,
    cublasHandle_t handle, cudaStream_t stream);

// release inverse_solver
linalgcuError_t fastect_inverse_solver_release(fastectInverseSolver_t* solverPointer);

// calc excitation
linalgcuError_t fastect_inverse_solver_calc_excitation(fastectInverseSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

// inverse solving
linalgcuError_t fastect_inverse_solver_solve(fastectInverseSolver_t solver,
    linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps, cublasHandle_t handle,
    cudaStream_t stream);

#endif
