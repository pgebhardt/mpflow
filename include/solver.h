// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_SOLVER_H
#define FASTECT_SOLVER_H

// solver struct
typedef struct {
    fastect_forward_solver_t forward_solver;
    fastect_inverse_solver_t inverse_solver;
    linalgcu_matrix_t sigma;
    linalgcu_matrix_t jacobian;
    linalgcu_matrix_t voltage_calculation;
    linalgcu_matrix_t calculated_voltage;
    linalgcu_matrix_t measured_voltage;
} fastect_solver_s;
typedef fastect_solver_s* fastect_solver_t;

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t drive_count,
    linalgcu_size_t measurment_count, linalgcu_matrix_t drive_pattern,
    linalgcu_matrix_t measurment_pattern, linalgcu_matrix_data_t sigma_0,
    cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcu_error_t fastect_solver_release(fastect_solver_t* solverPointer);

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, cublasHandle_t handle,
    cudaStream_t stream);

#endif
