// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_MODEL_H
#define FASTEIT_MODEL_H

// solver model struct
typedef struct {
    fasteitMesh_t mesh;
    fasteitElectrodes_t electrodes;
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
} fasteitModel_s;
typedef fasteitModel_s* fasteitModel_t;

// create model
linalgcuError_t fasteit_model_create(fasteitModel_t* modelPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, cublasHandle_t handle, cudaStream_t stream);

// release model
linalgcuError_t fasteit_model_release(fasteitModel_t* modelPointer);

// init system matrix 2D
linalgcuError_t fasteit_model_init_2D_system_matrix(fasteitModel_t self, cublasHandle_t handle,
    cudaStream_t stream);

// init residual matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_init_residual_matrix(fasteitModel_t self, linalgcuMatrix_t gamma,
    cudaStream_t stream);

// update system matrix
linalgcuError_t fasteit_model_update_system_matrices(fasteitModel_t self,
    linalgcuMatrix_t gamma, cublasHandle_t handle, cudaStream_t stream);

// update system matrix 2D
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_update_2D_system_matrix(fasteitModel_t self,
    linalgcuMatrix_t gamma, cudaStream_t stream);

// update residual matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_update_residual_matrix(fasteitModel_t,
    linalgcuMatrix_t gamma, cudaStream_t stream);

// init exitation matrix
linalgcuError_t fasteit_model_init_exitation_matrix(fasteitModel_t self,
    cudaStream_t stream);

// calc excitaions
linalgcuError_t fasteit_model_calc_excitaions(fasteitModel_t self, linalgcuMatrix_t* excitations,
    linalgcuMatrix_t pattern, cublasHandle_t handle, cudaStream_t stream);

#endif
