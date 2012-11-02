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
    linalgcuMatrix_t connectivityMatrix;
    linalgcuMatrix_t elementalSystemMatrix;
    linalgcuMatrix_t elementalResidualMatrix;
    linalgcuSize_t numHarmonics;
} fasteitModel_s;
typedef fasteitModel_s* fasteitModel_t;

// create model
linalgcuError_t fasteit_model_create(fasteitModel_t* modelPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, cublasHandle_t handle, cudaStream_t stream);

// release model
linalgcuError_t fasteit_model_release(fasteitModel_t* modelPointer);

// init model
linalgcuError_t fasteit_model_init(fasteitModel_t self, cublasHandle_t handle, cudaStream_t stream);

// init system matrix 2D
linalgcuError_t fasteit_model_init_sparse_matrices(fasteitModel_t self, cublasHandle_t handle,
    cudaStream_t stream);

// update model
linalgcuError_t fasteit_model_update(fasteitModel_t self, linalgcuMatrix_t gamma,
    cublasHandle_t handle, cudaStream_t stream);

// update matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_update_matrix(fasteitModel_t self,
    linalgcuSparseMatrix_t matrix, linalgcuMatrix_t elements, linalgcuMatrix_t gamma,
    cudaStream_t stream);

// init exitation matrix
linalgcuError_t fasteit_model_init_exitation_matrix(fasteitModel_t self,
    cudaStream_t stream);

// calc excitaions
linalgcuError_t fasteit_model_calc_excitaions(fasteitModel_t self, linalgcuMatrix_t* excitations,
    linalgcuMatrix_t pattern, cublasHandle_t handle, cudaStream_t stream);

// reduce matrix
LINALGCU_EXTERN_C
linalgcuError_t fasteit_model_reduce_matrix(fasteitModel_t self, linalgcuMatrix_t matrix,
    linalgcuMatrix_t intermediateMatrix, cudaStream_t stream);

#endif
