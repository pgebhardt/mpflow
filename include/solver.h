// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_SOLVER_H
#define FASTECT_SOLVER_H

// solver struct
typedef struct {
    fastect_forward_solver_t applied_solver;
    fastect_forward_solver_t lead_solver;
    fastect_inverse_solver_t inverse_solver;
    fastect_mesh_t mesh;
    fastect_electrodes_t electrodes;
    linalgcu_matrix_t jacobian;
    linalgcu_matrix_t voltage_calculation;
    linalgcu_matrix_t calculated_voltage;
    linalgcu_matrix_t measured_voltage;
    linalgcu_matrix_t sigma_ref;
} fastect_solver_s;
typedef fastect_solver_s* fastect_solver_t;

// create solver
linalgcu_error_t fastect_solver_create(fastect_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t drive_count,
    linalgcu_size_t measurment_count, linalgcu_matrix_t drive_pattern,
    linalgcu_matrix_t measurment_pattern, linalgcu_matrix_data_t sigma_ref,
    cublasHandle_t handle, cudaStream_t stream);

// create solver from config
linalgcu_error_t fastect_solver_from_config(fastect_solver_t* solverPointer,
    config_t* config, cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcu_error_t fastect_solver_release(fastect_solver_t* solverPointer);

// calc jacobian
LINALGCU_EXTERN_C
linalgcu_error_t fastect_solver_calc_jacobian(fastect_solver_t solver,
    cudaStream_t stream);

// forward solving
linalgcu_error_t fastect_solver_forward_solve(fastect_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream);

// solving
linalgcu_error_t fastect_solver_solve(fastect_solver_t solver, linalgcu_size_t linear_frames,
    cublasHandle_t handle, cudaStream_t stream);

#endif
