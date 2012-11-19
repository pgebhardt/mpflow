// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SOLVER_H
#define FASTEIT_SOLVER_H

// solver struct
typedef struct {
    fasteitForwardSolver_t forwardSolver;
    fasteitInverseSolver_t inverseSolver;
    linalgcuMatrix_t dGamma;
    linalgcuMatrix_t gamma;
    linalgcuMatrix_t measuredVoltage;
    linalgcuMatrix_t calibrationVoltage;
} fasteitSolver_s;
typedef fasteitSolver_s* fasteitSolver_t;

// create solver
linalgcuError_t fasteit_solver_create(fasteitSolver_t* solverPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrix_t measurmentPattern,
    linalgcuMatrix_t drivePattern, linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrixData_t numHarmonics, linalgcuMatrixData_t sigmaRef,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcuError_t fasteit_solver_release(fasteitSolver_t* solverPointer);

// pre solve for accurate initial jacobian
linalgcuError_t fasteit_solver_pre_solve(fasteitSolver_t self, cublasHandle_t handle,
    cudaStream_t stream);

// calibrate
linalgcuError_t fasteit_solver_calibrate(fasteitSolver_t self, cublasHandle_t handle,
    cudaStream_t stream);

// solving
linalgcuError_t fasteit_solver_solve(fasteitSolver_t self, cublasHandle_t handle,
    cudaStream_t stream);

#endif
