// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_CALIBRATION_SOLVER_H
#define FASTECT_CALIBRATION_SOLVER_H

// solver struct
typedef struct {
    fastectConjugateSolver_t conjugate_solver;
    linalgcuMatrix_t dVoltage;
    linalgcuMatrix_t dGamma;
    linalgcuMatrix_t zeros;
    linalgcuMatrix_t excitation;
    linalgcuMatrix_t systemMatrix;
    linalgcuMatrix_t jacobianSquare;
    linalgcuMatrix_t regularization;
    linalgcuMatrixData_t regularizationFactor;
} fastectCalibrationSolver_s;
typedef fastectCalibrationSolver_s* fastectCalibrationSolver_t;

// create calibration_solver
linalgcuError_t fastect_calibration_solver_create(fastectCalibrationSolver_t* solverPointer,
    linalgcuMatrix_t jacobian, linalgcuMatrixData_t regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream);

// release calibration_solver
linalgcuError_t fastect_calibration_solver_release(fastectCalibrationSolver_t* solverPointer);

// calc system matrix
linalgcuError_t fastect_calibration_solver_calc_system_matrix(
    fastectCalibrationSolver_t solver, linalgcuMatrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream);

// calc excitation
linalgcuError_t fastect_calibration_solver_calc_excitation(fastectCalibrationSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

// calibration solving
linalgcuError_t fastect_calibration_solver_calibrate(fastectCalibrationSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
    linalgcuMatrix_t measuredVoltage, linalgcuMatrix_t gamma,
    linalgcuSize_t steps, cublasHandle_t handle, cudaStream_t stream);

#endif
