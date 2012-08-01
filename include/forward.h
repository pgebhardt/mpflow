// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_FORWARD_SOLVER_H
#define FASTECT_FORWARD_SOLVER_H

// solver struct
typedef struct {
    fastect_grid_t grid;
    fastect_conjugate_solver_t conjugate_solver;
    linalgcu_size_t count;
    linalgcu_matrix_t phi;
    linalgcu_matrix_t* f;
} fastect_forward_solver_s;
typedef fastect_forward_solver_s* fastect_forward_solver_t;

// create forward_solver
linalgcu_error_t fastect_forward_solver_create(fastect_forward_solver_t* solverPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, linalgcu_size_t count,
    linalgcu_matrix_t drive_pattern, cublasHandle_t handle, cudaStream_t stream);

// release forward_solver
linalgcu_error_t fastect_forward_solver_release(fastect_forward_solver_t* solverPointer);

// calc excitaion
linalgcu_error_t fastect_forward_solver_calc_excitaion(fastect_forward_solver_t _solver,
    fastect_mesh_t mesh, linalgcu_matrix_t drive_pattern, cublasHandle_t handle,
    cudaStream_t stream);

// forward solving
linalgcu_error_t fastect_forward_solver_solve(fastect_forward_solver_t solver,
    cublasHandle_t handle, cudaStream_t stream);

#endif
