// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_CALIBRATION_SOLVER_H
#define FASTECT_CALIBRATION_SOLVER_H

// solver struct
typedef struct {
    fastect_conjugate_solver_t conjugate_solver;
    linalgcu_matrix_t dVoltage;
    linalgcu_matrix_t dSigma;
    linalgcu_matrix_t zeros;
    linalgcu_matrix_t excitation;
    linalgcu_matrix_t systemMatrix;
    linalgcu_matrix_t regularization;
    linalgcu_matrix_data_t regularizationFactor;
} fastect_calibration_solver_s;
typedef fastect_calibration_solver_s* fastect_calibration_solver_t;

// create calibration_solver
linalgcu_error_t fastect_calibration_solver_create(fastect_calibration_solver_t* solverPointer,
    linalgcu_matrix_t jacobian, linalgcu_matrix_data_t regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream);

// release calibration_solver
linalgcu_error_t fastect_calibration_solver_release(fastect_calibration_solver_t* solverPointer);

// calc system matrix
linalgcu_error_t fastect_calibration_solver_calc_system_matrix(
    fastect_calibration_solver_t solver, linalgcu_matrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream);

// calc excitation
linalgcu_error_t fastect_calibration_solver_calc_excitation(fastect_calibration_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

// calibration solving
linalgcu_error_t fastect_calibration_solver_calibrate(fastect_calibration_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, linalgcu_matrix_t sigma,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream);

#endif
