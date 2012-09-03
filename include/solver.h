// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_SOLVER_H
#define FASTECT_SOLVER_H

// solver struct
typedef struct {
    fastect_forward_solver_t forwardSolver;
    fastect_inverse_solver_t inverseSolver;
    linalgcu_matrix_t dSigma;
    linalgcu_matrix_t sigmaRef;
    linalgcu_matrix_t jacobian;
    linalgcu_matrix_t calculatedVoltage;
    linalgcu_matrix_t measuredVoltage;
    linalgcu_bool_t linearMode;
    cublasHandle_t cublasHandle;
} fastect_solver_s;
typedef fastect_solver_s* fastect_solver_t;

// solver config struct
typedef struct {
    linalgcu_size_t measurmentCount;
    linalgcu_size_t driveCount;
    linalgcu_matrix_t drivePattern;
    linalgcu_matrix_t measurmentPattern;
    fastect_mesh_t mesh;
    fastect_electrodes_t electrodes;
    linalgcu_matrix_data_t regularizationFactor;
    linalgcu_matrix_data_t sigma0;
    linalgcu_bool_t linearMode;
} fastect_solver_config_s;
typedef fastect_solver_config_s* fastect_solver_config_t;

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_solver_config_t config, cudaStream_t stream);

// release solver
linalgcu_error_t fastect_solver_release(fastect_solver_t* solverPointer);

// pre solve for accurate initial jacobian
linalgcu_error_t fastect_solver_pre_solve(fastect_solver_t solver, cudaStream_t stream);

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, cudaStream_t stream);

#endif
