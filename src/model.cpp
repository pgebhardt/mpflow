// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fasteit.h"

// create solver model
linalgcuError_t fasteit_model_create(fasteitModel_t* modelPointer,
    fasteitMesh_t mesh, fasteitElectrodes_t electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((modelPointer == NULL) || (mesh == NULL) || (electrodes == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init model pointer
    *modelPointer = NULL;

    // create model struct
    fasteitModel_t self = malloc(sizeof(fasteitModel_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->mesh = mesh;
    self->electrodes = electrodes;
    self->sigmaRef = sigmaRef;
    self->systemMatrix = NULL;
    self->SMatrix = NULL;
    self->RMatrix = NULL;
    self->excitationMatrix = NULL;
    self->connectivityMatrix = NULL;
    self->elementalSMatrix = NULL;
    self->elementalRMatrix = NULL;
    self->numHarmonics = numHarmonics;

    // create system matrices buffer
    self->systemMatrix = malloc(sizeof(linalgcuSparseMatrix_t) * (self->numHarmonics + 1));

    // check success
    if (self->systemMatrix == NULL) {
        // cleanup
        fasteit_model_release(&self);

        return LINALGCU_ERROR;
    }

    // create sparse matrices
    error = fasteit_model_create_sparse_matrices(self, handle, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_model_release(&self);

        return error;
    }

    // create matrices
    error  = linalgcu_matrix_create(&self->excitationMatrix,
        self->mesh->nodeCount, self->electrodes->count, stream);
    error |= linalgcu_matrix_create(&self->connectivityMatrix, self->mesh->nodeCount,
        LINALGCU_SPARSE_SIZE * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&self->elementalSMatrix, self->mesh->nodeCount,
        LINALGCU_SPARSE_SIZE * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&self->elementalRMatrix, self->mesh->nodeCount,
        LINALGCU_SPARSE_SIZE * LINALGCU_BLOCK_SIZE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_model_release(&self);

        return error;
    }

    // init model
    error |= fasteit_model_init(self, handle, stream);

    // init excitaion matrix
    error |= fasteit_model_init_exitation_matrix(self, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fasteit_model_release(&self);

        return error;
    }

    // set model pointer
    *modelPointer = self;

    return LINALGCU_SUCCESS;
}

// release solver model
linalgcuError_t fasteit_model_release(fasteitModel_t* modelPointer) {
    // check input
    if ((modelPointer == NULL) || (*modelPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get model
    fasteitModel_t self = *modelPointer;

    // cleanup
    fasteit_mesh_release(&self->mesh);
    fasteit_electrodes_release(&self->electrodes);
    linalgcu_sparse_matrix_release(&self->SMatrix);
    linalgcu_sparse_matrix_release(&self->RMatrix);
    linalgcu_matrix_release(&self->excitationMatrix);
    linalgcu_matrix_release(&self->connectivityMatrix);
    linalgcu_matrix_release(&self->elementalSMatrix);
    linalgcu_matrix_release(&self->elementalRMatrix);

    if (self->systemMatrix != NULL) {
        for (linalgcuSize_t i = 0; i < self->numHarmonics + 1; i++) {
            linalgcu_sparse_matrix_release(&self->systemMatrix[i]);
        }
        free(self->systemMatrix);
    }

    // free struct
    free(self);

    // set model pointer to NULL
    *modelPointer = NULL;

    return LINALGCU_SUCCESS;
}

// create sparse matrices
linalgcuError_t fasteit_model_create_sparse_matrices(fasteitModel_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calc initial system matrix
    // create matrices
    linalgcuMatrix_t systemMatrix;
    error = linalgcu_matrix_create(&systemMatrix,
        self->mesh->nodeCount, self->mesh->nodeCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    // calc generate empty system matrix
    linalgcuMatrixData_t id[FASTEIT_NODES_PER_ELEMENT];
    for (linalgcuSize_t k = 0; k < self->mesh->elementCount; k++) {
        // get nodes for element
        for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
            linalgcu_matrix_get_element(self->mesh->elements, &id[i], k, i);
        }

        // set system matrix elements
        for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
            for (linalgcuSize_t j = 0; j < FASTEIT_NODES_PER_ELEMENT; j++) {
                linalgcu_matrix_set_element(systemMatrix, 1.0f, (linalgcuSize_t)id[i],
                    (linalgcuSize_t)id[j]);
            }
        }
    }

    // copy matrices to device
    error = linalgcu_matrix_copy_to_device(systemMatrix, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_matrix_release(&systemMatrix);

        return LINALGCU_ERROR;
    }

    // create sparse matrices
    error  = linalgcu_sparse_matrix_create(&self->SMatrix, systemMatrix, stream);
    error |= linalgcu_sparse_matrix_create(&self->RMatrix, systemMatrix, stream);

    for (linalgcuSize_t i = 0; i < self->numHarmonics + 1; i++) {
        error |= linalgcu_sparse_matrix_create(&self->systemMatrix[i], systemMatrix, stream);
    }

    // cleanup
    linalgcu_matrix_release(&systemMatrix);

    return error;
}

// init model
linalgcuError_t fasteit_model_init(fasteitModel_t self, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // create intermediate matrices
    linalgcuMatrix_t elementCount, connectivityMatrix, elementalRMatrix, elementalSMatrix;
    error  = linalgcu_matrix_create(&elementCount, self->mesh->nodeCount,
        self->mesh->nodeCount, stream);
    error |= linalgcu_matrix_create(&connectivityMatrix, self->connectivityMatrix->rows,
        elementCount->columns * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&elementalSMatrix,
        self->elementalSMatrix->rows, elementCount->columns * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&elementalRMatrix,
        self->elementalRMatrix->rows, elementCount->columns * LINALGCU_BLOCK_SIZE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // init connectivityMatrix
    for (linalgcuSize_t i = 0; i < connectivityMatrix->rows; i++) {
        for (linalgcuSize_t j = 0; j < connectivityMatrix->columns; j++) {
            linalgcu_matrix_set_element(connectivityMatrix, -1.0f, i, j);
        }
    }
    for (linalgcuSize_t i = 0; i < self->connectivityMatrix->rows; i++) {
        for (linalgcuSize_t j = 0; j < self->connectivityMatrix->columns; j++) {
            linalgcu_matrix_set_element(self->connectivityMatrix, -1.0f, i, j);
        }
    }
    linalgcu_matrix_copy_to_device(self->connectivityMatrix, stream);

    // fill intermediate connectivity and elemental matrices
    linalgcuMatrixData_t id[FASTEIT_NODES_PER_ELEMENT],
        x[2 * FASTEIT_NODES_PER_ELEMENT], y[2 * FASTEIT_NODES_PER_ELEMENT];
    linalgcuMatrixData_t temp;
    fasteitBasis_t basis[FASTEIT_NODES_PER_ELEMENT];

    for (linalgcuSize_t k = 0; k < self->mesh->elementCount; k++) {
        // get nodes for element
        for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
            linalgcu_matrix_get_element(self->mesh->elements, &id[i], k, i);
            linalgcu_matrix_get_element(self->mesh->nodes, &x[i],
                (linalgcuSize_t)id[i], 0);
            linalgcu_matrix_get_element(self->mesh->nodes, &y[i],
                (linalgcuSize_t)id[i], 1);

            // get coordinates once more for permutations
            x[i + FASTEIT_NODES_PER_ELEMENT] = x[i];
            y[i + FASTEIT_NODES_PER_ELEMENT] = y[i];
        }

        // calc corresponding basis functions
        for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
            fasteit_basis_create(&basis[i], &x[i], &y[i]);
        }

        // set connectivity and elemental residual matrix elements
        for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
            for (linalgcuSize_t j = 0; j < FASTEIT_NODES_PER_ELEMENT; j++) {
                // get current element count
                linalgcu_matrix_get_element(elementCount, &temp,
                    (linalgcuSize_t)id[i], (linalgcuSize_t)id[j]);

                // set connectivity element
                linalgcu_matrix_set_element(connectivityMatrix,
                    (linalgcuMatrixData_t)k, (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // set elemental system element
                linalgcu_matrix_set_element(elementalSMatrix,
                    fasteit_basis_integrate_gradient_with_basis(basis[i], basis[j]),
                    (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // set elemental residual element
                linalgcu_matrix_set_element(elementalRMatrix,
                    fasteit_basis_integrate_with_basis(basis[i], basis[j]),
                    (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // increment element count
                elementCount->hostData[(linalgcuSize_t)id[i] + (linalgcuSize_t)id[j] *
                    elementCount->rows] += 1.0f;
            }
        }

        // cleanup
        for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
            fasteit_basis_release(&basis[i]);
        }
    }

    // upload intermediate matrices
    linalgcu_matrix_copy_to_device(connectivityMatrix, stream);
    linalgcu_matrix_copy_to_device(elementalSMatrix, stream);
    linalgcu_matrix_copy_to_device(elementalRMatrix, stream);

    // reduce matrices
    fasteit_model_reduce_matrix(self, self->connectivityMatrix, connectivityMatrix,
        self->SMatrix->density, stream);
    fasteit_model_reduce_matrix(self, self->elementalSMatrix, elementalSMatrix,
        self->SMatrix->density, stream);
    fasteit_model_reduce_matrix(self, self->elementalRMatrix, elementalRMatrix,
        self->SMatrix->density, stream);

    // create gamma
    linalgcuMatrix_t gamma;
    error  = linalgcu_matrix_create(&gamma, self->mesh->elementCount, 1, stream);

    // update matrices
    error |= fasteit_model_update_matrix(self, self->SMatrix, self->elementalSMatrix,
        gamma, stream);
    error |= fasteit_model_update_matrix(self, self->RMatrix, self->elementalRMatrix,
        gamma, stream);

    // cleanup
    linalgcu_matrix_release(&gamma);
    linalgcu_matrix_release(&elementCount);
    linalgcu_matrix_release(&connectivityMatrix);
    linalgcu_matrix_release(&elementalSMatrix);
    linalgcu_matrix_release(&elementalRMatrix);

    return error;
}

// update system matrix
linalgcuError_t fasteit_model_update(fasteitModel_t self, linalgcuMatrix_t gamma,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;
    cublasStatus_t cublasError = CUBLAS_STATUS_SUCCESS;

    // update 2d systemMatrix
    error  = fasteit_model_update_matrix(self, self->SMatrix, self->elementalSMatrix, gamma, stream);

    // update residual matrix
    error |= fasteit_model_update_matrix(self, self->RMatrix, self->elementalRMatrix, gamma, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // create system matrices for all harmonics
    linalgcuMatrixData_t alpha = 0.0f;
    for (linalgcuSize_t n = 0; n < self->numHarmonics + 1; n++) {
        // calc alpha
        alpha = (2.0f * n * M_PI / self->mesh->height) *
            (2.0f * n * M_PI / self->mesh->height);

        // init system matrix with 2d system matrix
        cublasError |= cublasScopy(handle, self->SMatrix->rows * LINALGCU_SPARSE_SIZE,
            self->SMatrix->values, 1, self->systemMatrix[n]->values, 1);

        // add alpha * residualMatrix
        cublasError |= cublasSaxpy(handle, self->SMatrix->rows * LINALGCU_SPARSE_SIZE, &alpha,
            self->RMatrix->values, 1, self->systemMatrix[n]->values, 1);
    }

    // check error
    if (cublasError != CUBLAS_STATUS_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
}

// init exitation matrix
linalgcuError_t fasteit_model_init_exitation_matrix(fasteitModel_t self,
    cudaStream_t stream) {
    // check input
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // fill exitation_matrix matrix
    linalgcuMatrixData_t id[FASTEIT_NODES_PER_EDGE];
    linalgcuMatrixData_t x[2 * FASTEIT_NODES_PER_EDGE], y[2 * FASTEIT_NODES_PER_EDGE];

    for (linalgcuSize_t i = 0; i < self->mesh->boundaryCount; i++) {
        for (linalgcuSize_t l = 0; l < self->electrodes->count; l++) {
            for (linalgcuSize_t k = 0; k < FASTEIT_NODES_PER_EDGE; k++) {
                // get node id
                linalgcu_matrix_get_element(self->mesh->boundary, &id[k], i, k);

                // get coordinates
                linalgcu_matrix_get_element(self->mesh->nodes, &x[k], (linalgcuSize_t)id[k], 0);
                linalgcu_matrix_get_element(self->mesh->nodes, &y[k], (linalgcuSize_t)id[k], 1);

                // set coordinates for permutations
                x[k + FASTEIT_NODES_PER_EDGE] = x[k];
                y[k + FASTEIT_NODES_PER_EDGE] = y[k];
            }

            // calc elements
            linalgcuMatrixData_t oldValue = 0.0f;
            for (linalgcuSize_t k = 0; k < FASTEIT_NODES_PER_EDGE; k++) {
                // get current value
                linalgcu_matrix_get_element(self->excitationMatrix, &oldValue, (linalgcuSize_t)id[k], l);

                // add new value
                linalgcu_matrix_set_element(self->excitationMatrix, oldValue - fasteit_basis_integrate_boundary_edge(
                    &x[k], &y[k], &self->electrodes->electrodesStart[l * 2], &self->electrodes->electrodesEnd[l * 2]) /
                    self->electrodes->width, (linalgcuSize_t)id[k], l);
            }
        }
    }

    // upload matrix
    linalgcu_matrix_copy_to_device(self->excitationMatrix, stream);

    return LINALGCU_SUCCESS;
}

// calc excitaion components
linalgcuError_t fasteit_model_calc_excitaion_components(fasteitModel_t self,
    linalgcuMatrix_t* component, linalgcuMatrix_t pattern, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if ((self == NULL) || (component == NULL) || (pattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calc excitation matrices
    for (linalgcuSize_t n = 0; n < self->numHarmonics + 1; n++) {
        // Run multiply once more to avoid cublas error
        linalgcu_matrix_multiply(component[n], self->excitationMatrix,
            pattern, handle, stream);
        error |= linalgcu_matrix_multiply(component[n], self->excitationMatrix,
            pattern, handle, stream);
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    error |= linalgcu_matrix_scalar_multiply(component[0],
        1.0f / self->mesh->height, stream);

    // calc harmonics
    for (linalgcuSize_t n = 1; n < self->numHarmonics + 1; n++) {
        error |= linalgcu_matrix_scalar_multiply(component[n],
            2.0f * sin(n * M_PI * self->electrodes->height / self->mesh->height) /
            (n * M_PI * self->electrodes->height), stream);
    }

    return error;
}
