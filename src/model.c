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

    // check success
    if (error != LINALGCU_SUCCESS) {
        return LINALGCU_ERROR;
    }

    return LINALGCU_SUCCESS;
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

    return LINALGCU_SUCCESS;
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

linalgcuMatrixData_t fasteit_model_angle(linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    if (x > 0.0f) {
        return atan(y / x);
    }
    else if ((x < 0.0f) && (y >= 0.0f)) {
        return atan(y / x) + M_PI;
    }
    else if ((x < 0.0f) && (y < 0.0f)) {
        return atan(y / x) - M_PI;
    }
    else if ((x == 0.0f) && (y > 0.0f)) {
        return M_PI / 2.0f;
    }
    else if ((x == 0.0f) && (y < 0.0f)) {
        return - M_PI / 2.0f;
    }
    else {
        return 0.0f;
    }
}

linalgcuMatrixData_t fasteit_model_integrate_basis(linalgcuMatrixData_t* start,
    linalgcuMatrixData_t* end, linalgcuMatrixData_t* electrodeStart,
    linalgcuMatrixData_t* electrodeEnd) {
    // integral
    linalgcuMatrixData_t integral = 0.0f;

    // calc radius
    linalgcuMatrixData_t radius = sqrt(electrodeStart[0] * electrodeStart[0]
        + electrodeStart[1] * electrodeStart[1]);

    // calc angle
    linalgcuMatrixData_t angleStart = fasteit_model_angle(start[0], start[1]);
    linalgcuMatrixData_t angleEnd = fasteit_model_angle(end[0], end[1]) - angleStart;
    linalgcuMatrixData_t angleElectrodeStart = fasteit_model_angle(electrodeStart[0], electrodeStart[1]) - angleStart;
    linalgcuMatrixData_t angleElectrodeEnd = fasteit_model_angle(electrodeEnd[0], electrodeEnd[1]) - angleStart;

    // correct angle
    angleEnd += (angleEnd < M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeStart += (angleElectrodeStart < M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeEnd += (angleElectrodeEnd < M_PI) ? 2.0f * M_PI : 0.0f;
    angleEnd -= (angleEnd > M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeStart -= (angleElectrodeStart > M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeEnd -= (angleElectrodeEnd > M_PI) ? 2.0f * M_PI : 0.0f;

    // calc parameter
    linalgcuMatrixData_t sEnd = radius * angleEnd;
    linalgcuMatrixData_t sElectrodeStart = radius * angleElectrodeStart;
    linalgcuMatrixData_t sElectrodeEnd = radius * angleElectrodeEnd;

    // integrate left triangle
    if (sEnd < 0.0f) {
        if ((sElectrodeStart < 0.0f) && (sElectrodeEnd > sEnd)) {
            if ((sElectrodeEnd >= 0.0f) && (sElectrodeStart <= sEnd)) {
                integral = -0.5f * sEnd;
            }
            else if ((sElectrodeEnd >= 0.0f) && (sElectrodeStart > sEnd)) {
                integral = -(sElectrodeStart - 0.5 * sElectrodeStart * sElectrodeStart / sEnd);
            }
            else if ((sElectrodeEnd < 0.0f) && (sElectrodeStart <= sEnd)) {
                integral = (sElectrodeEnd - 0.5 * sElectrodeEnd * sElectrodeEnd / sEnd) -
                           (sEnd - 0.5 * sEnd * sEnd / sEnd);
            }
            else if ((sElectrodeEnd < 0.0f) && (sElectrodeStart > sEnd)) {
                integral = (sElectrodeEnd - 0.5 * sElectrodeEnd * sElectrodeEnd / sEnd) -
                           (sElectrodeStart - 0.5 * sElectrodeStart * sElectrodeStart / sEnd);
            }
        }
    }
    else {
        // integrate right triangle
        if ((sElectrodeEnd > 0.0f) && (sEnd > sElectrodeStart)) {
            if ((sElectrodeStart <= 0.0f) && (sElectrodeEnd >= sEnd)) {
                integral = 0.5f * sEnd;
            }
            else if ((sElectrodeStart <= 0.0f) && (sElectrodeEnd < sEnd)) {
                integral = (sElectrodeEnd - 0.5f * sElectrodeEnd * sElectrodeEnd / sEnd);
            }
            else if ((sElectrodeStart > 0.0f) && (sElectrodeEnd >= sEnd)) {
                integral = (sEnd - 0.5f * sEnd * sEnd / sEnd) -
                            (sElectrodeStart - 0.5f * sElectrodeStart * sElectrodeStart / sEnd);
            }
            else if ((sElectrodeStart > 0.0f) && (sElectrodeEnd < sEnd)) {
                integral = (sElectrodeEnd - 0.5f * sElectrodeEnd * sElectrodeEnd / sEnd) -
                            (sElectrodeStart - 0.5f * sElectrodeStart * sElectrodeStart / sEnd);
            }
        }
    }

    return integral;
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
    linalgcuMatrixData_t id[2];
    linalgcuMatrixData_t node[2], end[2];

    for (int i = 0; i < self->mesh->boundaryCount; i++) {
        for (int j = 0; j < self->electrodes->count; j++) {
            // get boundary node id
            linalgcu_matrix_get_element(self->mesh->boundary, &id[0], i, 0);
            linalgcu_matrix_get_element(self->mesh->boundary, &id[1], i, 1);

            // get coordinates
            linalgcu_matrix_get_element(self->mesh->nodes, &node[0], (linalgcuSize_t)id[0], 0);
            linalgcu_matrix_get_element(self->mesh->nodes, &node[1], (linalgcuSize_t)id[0], 1);
            linalgcu_matrix_get_element(self->mesh->nodes, &end[0], (linalgcuSize_t)id[1], 0);
            linalgcu_matrix_get_element(self->mesh->nodes, &end[1], (linalgcuSize_t)id[1], 1);

            // calc element
            self->excitationMatrix->hostData[(linalgcuSize_t)id[0] + j * self->excitationMatrix->rows] -=
                fasteit_model_integrate_basis(node, end,
                    &self->electrodes->electrodesStart[j * 2],
                    &self->electrodes->electrodesEnd[j * 2]) / self->electrodes->width;
            self->excitationMatrix->hostData[(linalgcuSize_t)id[1] + j * self->excitationMatrix->rows] -=
                fasteit_model_integrate_basis(end, node,
                    &self->electrodes->electrodesStart[j * 2],
                    &self->electrodes->electrodesEnd[j * 2]) / self->electrodes->width;
        }
    }

    // upload matrix
    linalgcu_matrix_copy_to_device(self->excitationMatrix, stream);

    return LINALGCU_SUCCESS;
}

// calc excitaions
linalgcuError_t fasteit_model_calc_excitaions(fasteitModel_t self, linalgcuMatrix_t* excitations,
    linalgcuMatrix_t pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((self == NULL) || (excitations == NULL) || (pattern == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calc excitation matrices
    for (linalgcuSize_t n = 0; n < self->numHarmonics + 1; n++) {
        // Run multiply once more to avoid cublas error
        linalgcu_matrix_multiply(excitations[n], self->excitationMatrix,
            pattern, handle, stream);
        error |= linalgcu_matrix_multiply(excitations[n], self->excitationMatrix,
            pattern, handle, stream);
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    error |= linalgcu_matrix_scalar_multiply(excitations[0],
        1.0f / self->mesh->height, stream);

    // calc harmonics
    for (linalgcuSize_t n = 1; n < self->numHarmonics + 1; n++) {
        error |= linalgcu_matrix_scalar_multiply(excitations[n],
            2.0f * sin(n * M_PI * self->electrodes->height / self->mesh->height) /
            (n * M_PI * self->electrodes->height), stream);
    }

    return error;
}
