// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_FORWARD_SOLVER_H
#define FASTECT_FORWARD_SOLVER_H

// solver struct
typedef struct {
    fastectGrid_t grid;
    fastectConjugateSparseSolver_t driveSolver;
    fastectConjugateSparseSolver_t measurmentSolver;
    linalgcuMatrix_t* drivePhi;
    linalgcuMatrix_t* measurmentPhi;
    linalgcuMatrix_t* driveF;
    linalgcuMatrix_t* measurmentF;
    linalgcuMatrix_t voltageCalculation;
} fastectForwardSolver_s;
typedef fastectForwardSolver_s* fastectForwardSolver_t;

// create forward_solver
linalgcuError_t fastect_forward_solver_create(fastectForwardSolver_t* solverPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, linalgcuSize_t driveCount, linalgcuSize_t measurmentCount,
    linalgcuMatrix_t drivePattern, linalgcuMatrix_t measurmentPattern, cublasHandle_t handle,
    cudaStream_t stream);

// release forward_solver
linalgcuError_t fastect_forward_solver_release(fastectForwardSolver_t* solverPointer);

// calc jacobian
LINALGCU_EXTERN_C
linalgcuError_t fastect_forward_solver_calc_jacobian(fastectForwardSolver_t self,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t gamma, linalgcuSize_t harmonic,
    linalgcuBool_t additiv, cudaStream_t stream);

// forward solving
linalgcuError_t fastect_forward_solver_solve(fastectForwardSolver_t self,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t gamma, linalgcuMatrix_t voltage,
    linalgcuSize_t steps, cublasHandle_t handle, cudaStream_t stream);

#endif
