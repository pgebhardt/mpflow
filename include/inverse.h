// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_INVERSE_SOLVER_H
#define FASTECT_INVERSE_SOLVER_H

// solver struct
typedef struct {
    fastectConjugateSolver_t conjugateSolver;
    linalgcuMatrix_t dVoltage;
    linalgcuMatrix_t zeros;
    linalgcuMatrix_t excitation;
    linalgcuMatrix_t systemMatrix;
    linalgcuMatrix_t jacobianSquare;
    linalgcuMatrixData_t regularizationFactor;
} fastectInverseSolver_s;
typedef fastectInverseSolver_s* fastectInverseSolver_t;

// create inverse_solver
linalgcuError_t fastect_inverse_solver_create(fastectInverseSolver_t* solverPointer,
    linalgcuSize_t elementCount, linalgcuSize_t voltageCount,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream);

// release inverse_solver
linalgcuError_t fastect_inverse_solver_release(fastectInverseSolver_t* solverPointer);

// calc system matrix
linalgcuError_t fastect_inverse_solver_calc_system_matrix(
    fastectInverseSolver_t self, linalgcuMatrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream);

// calc excitation
linalgcuError_t fastect_inverse_solver_calc_excitation(fastectInverseSolver_t self,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

// inverse solving non linear
linalgcuError_t fastect_inverse_solver_non_linear(fastectInverseSolver_t self,
    linalgcuMatrix_t gamma, linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian,
    linalgcuMatrix_t calculatedVoltage, linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps,
    cublasHandle_t handle, cudaStream_t stream);

// inverse solving linear
linalgcuError_t fastect_inverse_solver_linear(fastectInverseSolver_t self,
    linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps, cublasHandle_t handle,
    cudaStream_t stream);

#endif
