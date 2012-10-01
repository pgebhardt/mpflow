// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_SOLVER_H
#define FASTECT_SOLVER_H

// solver struct
typedef struct {
    fastectForwardSolver_t forwardSolver;
    fastectInverseSolver_t inverseSolver;
    linalgcuMatrix_t dGamma;
    linalgcuMatrix_t gamma;
    linalgcuMatrix_t jacobian;
    linalgcuMatrix_t calculatedVoltage;
    linalgcuMatrix_t measuredVoltage;
    linalgcuMatrix_t calibrationVoltage;
} fastectSolver_s;
typedef fastectSolver_s* fastectSolver_t;

// create solver
linalgcuError_t fastect_solver_create(fastectSolver_t* solverPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuSize_t numHarmonics,
    linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrix_t measurmentPattern, linalgcuMatrix_t drivePattern,
    linalgcuMatrixData_t sigmaRef, linalgcuMatrixData_t regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcuError_t fastect_solver_release(fastectSolver_t* solverPointer);

// pre solve for accurate initial jacobian
linalgcuError_t fastect_solver_pre_solve(fastectSolver_t self, cublasHandle_t handle,
    cudaStream_t stream);

// calibrate
linalgcuError_t fastect_solver_calibrate(fastectSolver_t self, cublasHandle_t handle,
    cudaStream_t stream);

// solving
linalgcuError_t fastect_solver_solve(fastectSolver_t self, cublasHandle_t handle,
    cudaStream_t stream);

#endif
