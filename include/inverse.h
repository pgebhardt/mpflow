// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INVERSE_SOLVER_H
#define FASTEIT_INVERSE_SOLVER_H

// solver struct
typedef struct {
    fasteitConjugateSolver_t conjugateSolver;
    linalgcuMatrix_t dVoltage;
    linalgcuMatrix_t zeros;
    linalgcuMatrix_t excitation;
    linalgcuMatrix_t systemMatrix;
    linalgcuMatrix_t jacobianSquare;
    linalgcuMatrixData_t regularizationFactor;
} fasteitInverseSolver_s;
typedef fasteitInverseSolver_s* fasteitInverseSolver_t;

// create inverse_solver
linalgcuError_t fasteit_inverse_solver_create(fasteitInverseSolver_t* solverPointer,
    linalgcuSize_t elementCount, linalgcuSize_t voltageCount,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream);

// release inverse_solver
linalgcuError_t fasteit_inverse_solver_release(fasteitInverseSolver_t* solverPointer);

// calc system matrix
linalgcuError_t fasteit_inverse_solver_calc_system_matrix(
    fasteitInverseSolver_t self, linalgcuMatrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream);

// calc excitation
linalgcuError_t fasteit_inverse_solver_calc_excitation(fasteitInverseSolver_t self,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

// inverse solving non linear
linalgcuError_t fasteit_inverse_solver_non_linear(fasteitInverseSolver_t self,
    linalgcuMatrix_t gamma, linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian,
    linalgcuMatrix_t calculatedVoltage, linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps,
    cublasHandle_t handle, cudaStream_t stream);

// inverse solving linear
linalgcuError_t fasteit_inverse_solver_linear(fasteitInverseSolver_t self,
    linalgcuMatrix_t dGamma, linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, linalgcuSize_t steps, cublasHandle_t handle,
    cudaStream_t stream);

#endif
