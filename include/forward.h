// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_FORWARD_SOLVER_H
#define FASTECT_FORWARD_SOLVER_H

// c++ compatibility
#ifdef __cplusplus
extern "C" {
#endif

// solver struct
typedef struct {
    fastect_grid_t grid;
    fastect_conjugate_sparse_solver_t drive_solver;
    fastect_conjugate_sparse_solver_t measurment_solver;
    linalgcu_matrix_t drive_phi;
    linalgcu_matrix_t measurment_phi;
    linalgcu_matrix_t drive_f;
    linalgcu_matrix_t measurment_f;
    linalgcu_matrix_t voltage_calculation;
} fastect_forward_solver_s;
typedef fastect_forward_solver_s* fastect_forward_solver_t;

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t drive_count,
    linalgcu_size_t measurment_count, linalgcu_matrix_t drive_pattern,
    linalgcu_matrix_t measurment_pattern, cublasHandle_t handle, cudaStream_t stream);

// release forward_solver
linalgcu_error_t fastect_forward_solver_release(fastect_forward_solver_t* solverPointer);

// calc jacobian
LINALGCU_EXTERN_C
linalgcu_error_t fastect_forward_solver_calc_jacobian(fastect_forward_solver_t solver,
    linalgcu_matrix_t jacobian, cudaStream_t stream);

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    linalgcu_matrix_t sigma, linalgcu_matrix_t jacobian, linalgcu_matrix_t voltage,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
