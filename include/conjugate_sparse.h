// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_CONJUGATE_SPARSE_H
#define FASTECT_CONJUGATE_SPARSE_H

// conjugate solver struct
typedef struct {
    linalgcu_size_t rows;
    linalgcu_size_t columns;
    linalgcu_matrix_t residuum;
    linalgcu_matrix_t projection;
    linalgcu_matrix_t rsold;
    linalgcu_matrix_t rsnew;
    linalgcu_matrix_t tempVector;
    linalgcu_matrix_t tempNumber;
} fastect_conjugate_sparse_solver_s;
typedef fastect_conjugate_sparse_solver_s* fastect_conjugate_sparse_solver_t;

// create solver
linalgcu_error_t fastect_conjugate_sparse_solver_create(
    fastect_conjugate_sparse_solver_t* solverPointer, linalgcu_size_t rows,
    linalgcu_size_t columns, cublasHandle_t handle, cudaStream_t stream);

// release solver
linalgcu_error_t fastect_conjugate_sparse_solver_release(fastect_conjugate_sparse_solver_t* solverPointer);

// add scalar
LINALGCU_EXTERN_C
linalgcu_error_t fastect_conjugate_sparse_add_scalar(linalgcu_matrix_t vector,
    linalgcu_matrix_t scalar, linalgcu_size_t rows, linalgcu_size_t columns,
    cudaStream_t stream);

// update vector
LINALGCU_EXTERN_C
linalgcu_error_t fastect_conjugate_sparse_update_vector(linalgcu_matrix_t result,
    linalgcu_matrix_t x1, linalgcu_matrix_data_t sign, linalgcu_matrix_t x2,
    linalgcu_matrix_t r1, linalgcu_matrix_t r2, cudaStream_t stream);

// solve conjugate_sparse
linalgcu_error_t fastect_conjugate_sparse_solver_solve(fastect_conjugate_sparse_solver_t solver,
    linalgcu_sparse_matrix_t A, linalgcu_matrix_t x, linalgcu_matrix_t f,
    linalgcu_size_t iterations, cublasHandle_t handle, cudaStream_t stream);

// solve conjugate_sparse
linalgcu_error_t fastect_conjugate_sparse_solver_solve_regularized(
    fastect_conjugate_sparse_solver_t solver, linalgcu_sparse_matrix_t A, linalgcu_matrix_t x,
    linalgcu_matrix_t f, linalgcu_size_t iterations, cublasHandle_t handle, cudaStream_t stream);

#endif
