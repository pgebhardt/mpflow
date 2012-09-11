// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_FORWARD_SOLVER_H
#define FASTECT_FORWARD_SOLVER_H

// solver struct
typedef struct {
    fastect_grid_t grid;
    fastect_conjugate_sparse_solver_t driveSolver;
    fastect_conjugate_sparse_solver_t measurmentSolver;
    linalgcu_matrix_t* drivePhi;
    linalgcu_matrix_t* measurmentPhi;
    linalgcu_matrix_t* driveF;
    linalgcu_matrix_t* measurmentF;
    linalgcu_matrix_t voltageCalculation;
} fastect_forward_solver_s;
typedef fastect_forward_solver_s* fastect_forward_solver_t;

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_matrix_data_t sigmaRef,
    linalgcu_size_t numHarmonics, linalgcu_size_t driveCount, linalgcu_size_t measurmentCount,
    linalgcu_matrix_t drivePattern, linalgcu_matrix_t measurmentPattern, cublasHandle_t handle,
    cudaStream_t stream);

// release forward_solver
linalgcu_error_t fastect_forward_solver_release(fastect_forward_solver_t* solverPointer);

// calc jacobian
LINALGCU_EXTERN_C
linalgcu_error_t fastect_forward_solver_calc_jacobian(fastect_forward_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t gamma, linalgcu_matrix_data_t sigmaRef,
    linalgcu_size_t harmonic, linalgcu_bool_t additiv, cudaStream_t stream);

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t gamma, linalgcu_matrix_data_t sigmaRef,
    linalgcu_matrix_t voltage, linalgcu_size_t steps, cublasHandle_t handle,
    cudaStream_t stream);

#endif
