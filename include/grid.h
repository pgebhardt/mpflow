// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_GRID_H
#define FASTECT_GRID_H

// c++ compatibility
#ifdef __cplusplus
extern "C" {
#endif

// solver grid struct
typedef struct {
    fastect_mesh_t mesh;
    fastect_electrodes_t electrodes;
    linalgcu_sparse_matrix_t system_matrix;
    linalgcu_matrix_t excitation_matrix;
    linalgcu_sparse_matrix_t gradient_matrix_sparse;
    linalgcu_sparse_matrix_t gradient_matrix_transposed_sparse;
    linalgcu_matrix_t gradient_matrix_transposed;
    linalgcu_matrix_t area;
} fastect_grid_s;
typedef fastect_grid_s* fastect_grid_t;

// create grid
linalgcu_error_t fastect_grid_create(fastect_grid_t* gridPointer,
    fastect_mesh_t mesh, fastect_electrodes_t electrodes, cublasHandle_t handle,
    cudaStream_t stream);

// release grid
linalgcu_error_t fastect_grid_release(fastect_grid_t* gridPointer);

// init system matrix
linalgcu_error_t fastect_grid_init_system_matrix(fastect_grid_t grid, cublasHandle_t handle,
    cudaStream_t stream);

// update system matrix
LINALGCU_EXTERN_C
linalgcu_error_t fastect_grid_update_system_matrix(fastect_grid_t grid,
    linalgcu_matrix_t sigma, cudaStream_t stream);

// init exitation matrix
linalgcu_error_t fastect_grid_init_exitation_matrix(fastect_grid_t grid, cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
