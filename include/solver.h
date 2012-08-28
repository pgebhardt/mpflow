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
    linalgcu_matrix_t sigma_ref;
    linalgcu_matrix_t jacobian;
    linalgcu_matrix_t calculated_voltage;
    linalgcu_matrix_t measured_voltage;
    cublasHandle_t cublas_handle;
} fastect_solver_s;
typedef fastect_solver_s* fastect_solver_t;

// solver config struct
typedef struct {
    linalgcu_size_t measurment_count;
    linalgcu_size_t drive_count;
    linalgcu_matrix_t drive_pattern;
    linalgcu_matrix_t measurment_pattern;
    fastect_mesh_t mesh;
    fastect_electrodes_t electrodes;
    linalgcu_matrix_data_t regularization_factor;
    linalgcu_matrix_data_t sigma_0;
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
