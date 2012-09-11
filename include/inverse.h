// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_INVERSE_SOLVER_H
#define FASTECT_INVERSE_SOLVER_H

// solver struct
typedef struct {
    fastect_conjugate_solver_t conjugate_solver;
    linalgcu_matrix_t deltaVoltage;
    linalgcu_matrix_t zeros;
    linalgcu_matrix_t excitation;
    linalgcu_matrix_t systemMatrix;
} fastect_inverse_solver_s;
typedef fastect_inverse_solver_s* fastect_inverse_solver_t;

// create inverse_solver
linalgcu_error_t fastect_inverse_solver_create(fastect_inverse_solver_t* solverPointer,
    linalgcu_matrix_t systemMatrix, linalgcu_matrix_t jacobian, cublasHandle_t handle,
    cudaStream_t stream);

// release inverse_solver
linalgcu_error_t fastect_inverse_solver_release(fastect_inverse_solver_t* solverPointer);

// calc excitation
linalgcu_error_t fastect_inverse_solver_calc_excitation(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculatedVoltage,
    linalgcu_matrix_t measuredVoltage, linalgcu_matrix_t gamma,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream);

#endif
