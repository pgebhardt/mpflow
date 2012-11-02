// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_FORWARD_SOLVER_H
#define FASTEIT_FORWARD_SOLVER_H

// solver struct
typedef struct {
    fasteitModel_t model;
    fasteitSparseConjugateSolver_t conjugateSolver;
    linalgcuSize_t driveCount;
    linalgcuSize_t measurmentCount;
    linalgcuMatrix_t jacobian;
    linalgcuMatrix_t voltage;
    linalgcuMatrix_t* phi;
    linalgcuMatrix_t* excitation;
    linalgcuMatrix_t voltageCalculation;
    linalgcuMatrix_t area;
    linalgcuSparseMatrix_t gradientMatrixSparse;
} fasteitForwardSolver_s;
typedef fasteitForwardSolver_s* fasteitForwardSolver_t;

// create forward_solver
linalgcuError_t fasteit_forward_solver_create(fasteitForwardSolver_t* solverPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrix_t measurmentPattern,
    linalgcuMatrix_t drivePattern, linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrixData_t numHarmonics, linalgcuMatrixData_t sigmaRef, cublasHandle_t handle,
    cudaStream_t stream);

// release forward_solver
linalgcuError_t fasteit_forward_solver_release(fasteitForwardSolver_t* solverPointer);

// init jacobian calculation matrices
linalgcuError_t fasteit_forward_init_jacobian_calculation_matrices(fasteitForwardSolver_t self,
    cublasHandle_t handle, cudaStream_t stream);

// calc jacobian
LINALGCU_EXTERN_C
linalgcuError_t fasteit_forward_solver_calc_jacobian(fasteitForwardSolver_t self,
    linalgcuMatrix_t gamma, linalgcuSize_t harmonic, linalgcuBool_t additiv,
    cudaStream_t stream);

// forward solving
linalgcuError_t fasteit_forward_solver_solve(fasteitForwardSolver_t self,
    linalgcuMatrix_t gamma, linalgcuSize_t steps, cublasHandle_t handle, cudaStream_t stream);

#endif
