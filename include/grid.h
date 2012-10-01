// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_GRID_H
#define FASTECT_GRID_H

// solver grid struct
typedef struct {
    fastectMesh_t mesh;
    fastectElectrodes_t electrodes;
    linalgcuMatrixData_t sigmaRef;
    linalgcuSparseMatrix_t* systemMatrices;
    linalgcuSparseMatrix_t systemMatrix2D;
    linalgcuSparseMatrix_t residualMatrix;
    linalgcuMatrix_t excitationMatrix;
    linalgcuMatrix_t gradientMatrixTransposed;
    linalgcuSparseMatrix_t gradientMatrixTransposedSparse;
    linalgcuSparseMatrix_t gradientMatrixSparse;
    linalgcuMatrix_t connectivityMatrix;
    linalgcuMatrix_t elementalResidualMatrix;
    linalgcuMatrix_t area;
    linalgcuSize_t numHarmonics;
} fastectGrid_s;
typedef fastectGrid_s* fastectGrid_t;

// create grid
linalgcuError_t fastect_grid_create(fastectGrid_t* gridPointer,
    fastectMesh_t mesh, fastectElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, cublasHandle_t handle, cudaStream_t stream);

// release grid
linalgcuError_t fastect_grid_release(fastectGrid_t* gridPointer);

// init system matrix 2D
linalgcuError_t fastect_grid_init_2D_system_matrix(fastectGrid_t self, cublasHandle_t handle,
    cudaStream_t stream);

// init residual matrix
LINALGCU_EXTERN_C
linalgcuError_t fastect_grid_init_residual_matrix(fastectGrid_t self, linalgcuMatrix_t gamma,
    cudaStream_t stream);

// update system matrix
linalgcuError_t fastect_grid_update_system_matrices(fastectGrid_t self,
    linalgcuMatrix_t gamma, cublasHandle_t handle, cudaStream_t stream);

// update system matrix 2D
LINALGCU_EXTERN_C
linalgcuError_t fastect_grid_update_2D_system_matrix(fastectGrid_t self,
    linalgcuMatrix_t gamma, cudaStream_t stream);

// update residual matrix
LINALGCU_EXTERN_C
linalgcuError_t fastect_grid_update_residual_matrix(fastectGrid_t,
    linalgcuMatrix_t gamma, cudaStream_t stream);

// init exitation matrix
linalgcuError_t fastect_grid_init_exitation_matrix(fastectGrid_t self,
    cudaStream_t stream);

#endif
