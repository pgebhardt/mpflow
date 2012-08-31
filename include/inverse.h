// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_INVERSE_SOLVER_H
#define FASTECT_INVERSE_SOLVER_H

// solver struct
typedef struct {
    fastect_conjugate_solver_t conjugate_solver;
    linalgcu_matrix_t dU;
    linalgcu_matrix_t zeros;
    linalgcu_matrix_t f;
    linalgcu_matrix_t A;
    linalgcu_matrix_t regularization;
    linalgcu_matrix_data_t regularization_factor;
} fastect_inverse_solver_s;
typedef fastect_inverse_solver_s* fastect_inverse_solver_t;

// create inverse_solver
linalgcu_error_t fastect_inverse_solver_create(fastect_inverse_solver_t* solverPointer,
    linalgcu_matrix_t jacobian, linalgcu_matrix_data_t regularization_factor,
    cublasHandle_t handle, cudaStream_t stream);

// release inverse_solver
linalgcu_error_t fastect_inverse_solver_release(fastect_inverse_solver_t* solverPointer);

// calc system matrix
linalgcu_error_t fastect_inverse_solver_calc_system_matrix(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, cublasHandle_t handle, cudaStream_t stream);

// calc excitation
linalgcu_error_t fastect_inverse_solver_calc_excitation(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculated_voltage,
    linalgcu_matrix_t measured_voltage, cublasHandle_t handle, cudaStream_t stream);

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve_non_linear(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculated_voltage,
    linalgcu_matrix_t measured_voltage, linalgcu_matrix_t sigma,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream);

// inverse solving
linalgcu_error_t fastect_inverse_solver_solve_linear(fastect_inverse_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_matrix_t calculated_voltage,
    linalgcu_matrix_t measured_voltage, linalgcu_matrix_t sigma,
    linalgcu_size_t steps, cublasHandle_t handle, cudaStream_t stream);

#endif
