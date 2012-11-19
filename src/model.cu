// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <iostream>
#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create solver model
template <class BasisFunction>
Model<BasisFunction>::Model(Mesh* mesh, Electrodes* electrodes, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t numHarmonics, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("Model<BasisFunction>::Model: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw invalid_argument("Model<BasisFunction>::Model: electrodes == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Model<BasisFunction>::Model: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init member
    this->mMesh = mesh;
    this->mElectrodes = electrodes;
    this->mSigmaRef = sigmaRef;
    this->mSystemMatrix = NULL;
    this->mSMatrix = NULL;
    this->mRMatrix = NULL;
    this->mExcitationMatrix = NULL;
    this->mConnectivityMatrix = NULL;
    this->mElementalSMatrix = NULL;
    this->mElementalRMatrix = NULL;
    this->mNumHarmonics = numHarmonics;

    // create system matrices buffer
    this->mSystemMatrix = new linalgcuSparseMatrix_t[this->mNumHarmonics + 1];

    // create sparse matrices
    this->create_sparse_matrices(handle, stream);

    // create matrices
    error  = linalgcu_matrix_create(&this->mExcitationMatrix,
        this->mesh()->nodeCount(), this->mElectrodes->count(), stream);
    error |= linalgcu_matrix_create(&this->mConnectivityMatrix, this->mesh()->nodeCount(),
        LINALGCU_SPARSE_SIZE * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&this->mElementalSMatrix, this->mesh()->nodeCount(),
        LINALGCU_SPARSE_SIZE * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&this->mElementalRMatrix, this->mesh()->nodeCount(),
        LINALGCU_SPARSE_SIZE * LINALGCU_BLOCK_SIZE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Model<BasisFunction>::Model: create matrices");
    }

    // init model
    this->init(handle, stream);

    // init excitaion matrix
    this->init_excitation_matrix(stream);
}

// release solver model
template <class BasisFunction>
Model<BasisFunction>::~Model() {
    // cleanup
    if (this->mMesh != NULL) {
        delete this->mMesh;
    }
    if (this->mElectrodes != NULL) {
        delete this->mElectrodes;
    }
    linalgcu_sparse_matrix_release(&this->mSMatrix);
    linalgcu_sparse_matrix_release(&this->mRMatrix);
    linalgcu_matrix_release(&this->mExcitationMatrix);
    linalgcu_matrix_release(&this->mConnectivityMatrix);
    linalgcu_matrix_release(&this->mElementalSMatrix);
    linalgcu_matrix_release(&this->mElementalRMatrix);

    if (this->mSystemMatrix != NULL) {
        for (linalgcuSize_t i = 0; i < this->mNumHarmonics + 1; i++) {
            linalgcu_sparse_matrix_release(&this->mSystemMatrix[i]);
        }
        delete [] this->mSystemMatrix;
    }
}


// create sparse matrices
template <class BasisFunction>
void Model<BasisFunction>::create_sparse_matrices(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Model<BasisFunction>::create_sparse_matrices: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calc initial system matrix
    // create matrices
    linalgcuMatrix_t systemMatrix;
    error = linalgcu_matrix_create(&systemMatrix,
        this->mesh()->nodeCount(), this->mesh()->nodeCount(), stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Model<BasisFunction>::create_sparse_matrices: create matrices");
    }

    // calc generate empty system matrix
    linalgcuMatrixData_t id[BasisFunction::nodesPerElement];
    for (linalgcuSize_t k = 0; k < this->mesh()->elementCount(); k++) {
        // get nodes for element
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            linalgcu_matrix_get_element(this->mesh()->elements(), &id[i], k, i);
        }

        // set system matrix elements
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            for (linalgcuSize_t j = 0; j < BasisFunction::nodesPerElement; j++) {
                linalgcu_matrix_set_element(systemMatrix, 1.0f, (linalgcuSize_t)id[i],
                    (linalgcuSize_t)id[j]);
            }
        }
    }

    // copy matrix to device
    error = linalgcu_matrix_copy_to_device(systemMatrix, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Model<BasisFunction>::create_sparse_matrices: copy matrix to device");
    }

    // create sparse matrices
    error  = linalgcu_sparse_matrix_create(&this->mSMatrix, systemMatrix, stream);
    error |= linalgcu_sparse_matrix_create(&this->mRMatrix, systemMatrix, stream);

    for (linalgcuSize_t i = 0; i < this->numHarmonics() + 1; i++) {
        error |= linalgcu_sparse_matrix_create(&this->mSystemMatrix[i], systemMatrix, stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Model<BasisFunction>::create_sparse_matrices: create sparse matrces");
    }
}

// init model
template <class BasisFunction>
void Model<BasisFunction>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Model<BasisFunction>::init: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // create intermediate matrices
    linalgcuMatrix_t elementCount, connectivityMatrix, elementalRMatrix, elementalSMatrix;
    error  = linalgcu_matrix_create(&elementCount, this->mesh()->nodeCount(),
        this->mesh()->nodeCount(), stream);
    error |= linalgcu_matrix_create(&connectivityMatrix, this->mConnectivityMatrix->rows,
        elementCount->columns * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&elementalSMatrix,
        this->mElementalSMatrix->rows, elementCount->columns * LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&elementalRMatrix,
        this->mElementalRMatrix->rows, elementCount->columns * LINALGCU_BLOCK_SIZE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Model<BasisFunction>::init: create intermediate matrices");
    }

    // init connectivityMatrix
    for (linalgcuSize_t i = 0; i < connectivityMatrix->rows; i++) {
        for (linalgcuSize_t j = 0; j < connectivityMatrix->columns; j++) {
            linalgcu_matrix_set_element(connectivityMatrix, -1.0f, i, j);
        }
    }
    for (linalgcuSize_t i = 0; i < this->mConnectivityMatrix->rows; i++) {
        for (linalgcuSize_t j = 0; j < this->mConnectivityMatrix->columns; j++) {
            linalgcu_matrix_set_element(this->mConnectivityMatrix, -1.0f, i, j);
        }
    }
    linalgcu_matrix_copy_to_device(this->mConnectivityMatrix, stream);

    // fill intermediate connectivity and elemental matrices
    linalgcuMatrixData_t id[BasisFunction::nodesPerElement],
        x[BasisFunction::nodesPerElement * 2], y[BasisFunction::nodesPerElement * 2];
    BasisFunction* basis[BasisFunction::nodesPerElement];
    linalgcuMatrixData_t temp;

    for (linalgcuSize_t k = 0; k < this->mesh()->elementCount(); k++) {
        // get nodes for element
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            linalgcu_matrix_get_element(this->mesh()->elements(), &id[i], k, i);
            linalgcu_matrix_get_element(this->mesh()->nodes(), &x[i],
                (linalgcuSize_t)id[i], 0);
            linalgcu_matrix_get_element(this->mesh()->nodes(), &y[i],
                (linalgcuSize_t)id[i], 1);

            // get coordinates once more for permutations
            x[i + BasisFunction::nodesPerElement] = x[i];
            y[i + BasisFunction::nodesPerElement] = y[i];
        }

        // calc corresponding basis functions
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            basis[i] = new BasisFunction(&x[i], &y[i]);
        }

        // set connectivity and elemental residual matrix elements
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            for (linalgcuSize_t j = 0; j < BasisFunction::nodesPerElement; j++) {
                // get current element count
                linalgcu_matrix_get_element(elementCount, &temp,
                    (linalgcuSize_t)id[i], (linalgcuSize_t)id[j]);

                // set connectivity element
                linalgcu_matrix_set_element(connectivityMatrix,
                    (linalgcuMatrixData_t)k, (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // set elemental system element
                linalgcu_matrix_set_element(elementalSMatrix,
                    basis[i]->integrate_gradient_with_basis(*basis[j]),
                    (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // set elemental residual element
                linalgcu_matrix_set_element(elementalRMatrix,
                    basis[i]->integrate_with_basis(*basis[j]),
                    (linalgcuSize_t)id[i],
                    (linalgcuSize_t)(id[j] + connectivityMatrix->rows * temp));

                // increment element count
                elementCount->hostData[(linalgcuSize_t)id[i] + (linalgcuSize_t)id[j] *
                    elementCount->rows] += 1.0f;
            }
        }

        // cleanup
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            delete basis[i];
        }
    }

    // upload intermediate matrices
    linalgcu_matrix_copy_to_device(connectivityMatrix, stream);
    linalgcu_matrix_copy_to_device(elementalSMatrix, stream);
    linalgcu_matrix_copy_to_device(elementalRMatrix, stream);

    // reduce matrices
    this->reduce_matrix(this->mConnectivityMatrix, connectivityMatrix,
        this->mSMatrix->density, stream);
    this->reduce_matrix(this->mElementalSMatrix, elementalSMatrix,
        this->mSMatrix->density, stream);
    this->reduce_matrix(this->mElementalRMatrix, elementalRMatrix,
        this->mSMatrix->density, stream);

    // create gamma
    linalgcuMatrix_t gamma;
    error  = linalgcu_matrix_create(&gamma, this->mesh()->elementCount(), 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Model<BasisFunction>::init: create gamma");
    }

    // update matrices
    this->update_matrix(this->mSMatrix, this->mElementalSMatrix,
        gamma, stream);
    this->update_matrix(this->mRMatrix, this->mElementalRMatrix,
        gamma, stream);

    // cleanup
    linalgcu_matrix_release(&gamma);
    linalgcu_matrix_release(&elementCount);
    linalgcu_matrix_release(&connectivityMatrix);
    linalgcu_matrix_release(&elementalSMatrix);
    linalgcu_matrix_release(&elementalRMatrix);
}

// update system matrix
template <class BasisFunction>
void Model<BasisFunction>::update(linalgcuMatrix_t gamma, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("Model<BasisFunction>::update: gamma == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Model<BasisFunction>::init: handle == NULL");
    }

    // error
    cublasStatus_t cublasError = CUBLAS_STATUS_SUCCESS;

    // update 2d systemMatrix
    this->update_matrix(this->mSMatrix, this->mElementalSMatrix, gamma, stream);

    // update residual matrix
    this->update_matrix(this->mRMatrix, this->mElementalRMatrix, gamma, stream);

    // set cublas stream
    cublasSetStream(handle, stream);

    // create system matrices for all harmonics
    linalgcuMatrixData_t alpha = 0.0f;
    for (linalgcuSize_t n = 0; n < this->numHarmonics() + 1; n++) {
        // calc alpha
        alpha = (2.0f * n * M_PI / this->mesh()->height()) *
            (2.0f * n * M_PI / this->mesh()->height());

        // init system matrix with 2d system matrix
        cublasError = cublasScopy(handle, this->mSMatrix->rows * LINALGCU_SPARSE_SIZE,
            this->mSMatrix->values, 1, this->systemMatrix(n)->values, 1);

        // check error
        if (cublasError != CUBLAS_STATUS_SUCCESS) {
            throw logic_error(
                "Model<BasisFunction>::update: calc system matrices for all harmonics");
        }

        // add alpha * residualMatrix
        cublasError = cublasSaxpy(handle, this->mSMatrix->rows * LINALGCU_SPARSE_SIZE, &alpha,
            this->mRMatrix->values, 1, this->systemMatrix(n)->values, 1);

        // check error
        if (cublasError != CUBLAS_STATUS_SUCCESS) {
            throw logic_error(
                "Model<BasisFunction>::update: calc system matrices for all harmonics");
        }

    }
}

// init exitation matrix
template <class BasisFunction>
void Model<BasisFunction>::init_excitation_matrix(cudaStream_t stream) {
    // fill exitation_matrix matrix
    linalgcuMatrixData_t id[BasisFunction::nodesPerEdge],
        x[BasisFunction::nodesPerEdge * 2], y[BasisFunction::nodesPerEdge * 2];

    for (linalgcuSize_t i = 0; i < this->mesh()->boundaryCount(); i++) {
        for (linalgcuSize_t l = 0; l < this->mElectrodes->count(); l++) {
            for (linalgcuSize_t k = 0; k < BasisFunction::nodesPerEdge; k++) {
                // get node id
                linalgcu_matrix_get_element(this->mesh()->boundary(), &id[k], i, k);

                // get coordinates
                linalgcu_matrix_get_element(this->mesh()->nodes(), &x[k], (linalgcuSize_t)id[k], 0);
                linalgcu_matrix_get_element(this->mesh()->nodes(), &y[k], (linalgcuSize_t)id[k], 1);

                // set coordinates for permutations
                x[k + BasisFunction::nodesPerEdge] = x[k];
                y[k + BasisFunction::nodesPerEdge] = y[k];
            }

            // calc elements
            linalgcuMatrixData_t oldValue = 0.0f;
            for (linalgcuSize_t k = 0; k < BasisFunction::nodesPerEdge; k++) {
                // get current value
                linalgcu_matrix_get_element(this->mExcitationMatrix, &oldValue,
                    (linalgcuSize_t)id[k], l);

                // add new value
                linalgcu_matrix_set_element(this->mExcitationMatrix,
                    oldValue - BasisFunction::integrate_boundary_edge(
                        &x[k], &y[k], &this->mElectrodes->electrodesStart()[l * 2],
                        &this->mElectrodes->electrodesEnd()[l * 2]) /
                    this->mElectrodes->width(), (linalgcuSize_t)id[k], l);
            }
        }
    }

    // upload matrix
    linalgcu_matrix_copy_to_device(this->mExcitationMatrix, stream);
}

// calc excitaion components
template <class BasisFunction>
void Model<BasisFunction>::calc_excitation_components(linalgcuMatrix_t* component,
    linalgcuMatrix_t pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (component == NULL) {
        throw invalid_argument(
            "Model<BasisFunction>::calc_excitation_components: component == NULL");
    }
    if (pattern == NULL) {
        throw invalid_argument("Model<BasisFunction>::calc_excitation_components: pattern == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Model<BasisFunction>::calc_excitation_components: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // calc excitation matrices
    for (linalgcuSize_t n = 0; n < this->numHarmonics() + 1; n++) {
        // Run multiply once more to avoid cublas error
        linalgcu_matrix_multiply(component[n], this->mExcitationMatrix,
            pattern, handle, stream);
        error |= linalgcu_matrix_multiply(component[n], this->mExcitationMatrix,
            pattern, handle, stream);
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    error |= linalgcu_matrix_scalar_multiply(component[0],
        1.0f / this->mesh()->height(), stream);

    // calc harmonics
    for (linalgcuSize_t n = 1; n < this->numHarmonics() + 1; n++) {
        error |= linalgcu_matrix_scalar_multiply(component[n],
            2.0f * sin(n * M_PI * this->electrodes()->height() / this->mesh()->height()) /
            (n * M_PI * this->electrodes()->height()), stream);
    }

    // check error
    if (error != LINALGCU_SUCCESS) {
        throw logic_error(
            "Model<BasisFunction>::calc_excitation_components: calc fourier coefficients");
    }
}

// reduce connectivity and elementalResidual matrix
__global__ void reduce_matrix_kernel(linalgcuMatrixData_t* matrix,
    linalgcuMatrixData_t* intermediateMatrix, linalgcuColumnId_t* systemMatrixColumnIds,
    linalgcuSize_t rows, linalgcuSize_t density) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    linalgcuColumnId_t columnId = systemMatrixColumnIds[row * LINALGCU_SPARSE_SIZE + column];

    // check column id
    if (columnId == -1) {
        return;
    }

    // reduce matrices
    for (int k = 0; k < density; k++) {
        matrix[row + (column + k * LINALGCU_SPARSE_SIZE) * rows] =
            intermediateMatrix[row + (columnId + k * rows) * rows];
    }
}

// reduce matrix
template <class BasisFunction>
void Model<BasisFunction>::reduce_matrix(linalgcuMatrix_t matrix,
    linalgcuMatrix_t intermediateMatrix, linalgcuSize_t density, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("matrix == NULL");
    }
    if (intermediateMatrix == NULL) {
        throw invalid_argument("intermediateMatrix == NULL");
    }

    // block size
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE, 1);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // reduce matrix
    reduce_matrix_kernel<<<blocks, threads, 0, stream>>>(
        matrix->deviceData, intermediateMatrix->deviceData,
        this->mSMatrix->columnIds, matrix->rows,
        density);
}

// update matrix kernel
__global__ void update_matrix_kernel(linalgcuMatrixData_t* matrixValues,
    linalgcuColumnId_t* matrixColumnIds, linalgcuColumnId_t* columnIds,
    linalgcuMatrixData_t* connectivityMatrix, linalgcuMatrixData_t* elementalMatrix,
    linalgcuMatrixData_t* gamma, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t rows, linalgcuSize_t density) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    linalgcuMatrixData_t value = 0.0f;
    linalgcuColumnId_t elementId = -1;
    for (int k = 0; k < density; k++) {
        // get element id
        elementId = (linalgcuColumnId_t)connectivityMatrix[row +
            (column + k * LINALGCU_SPARSE_SIZE) * rows];

        value += elementId != -1 ? elementalMatrix[row +
            (column + k * LINALGCU_SPARSE_SIZE) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrixValues[row * LINALGCU_SPARSE_SIZE + column] = value;
}

// update matrix
template <class BasisFunction>
void Model<BasisFunction>::update_matrix(linalgcuSparseMatrix_t matrix,
    linalgcuMatrix_t elementalMatrix, linalgcuMatrix_t gamma, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("matrix == NULL");
    }
    if (elementalMatrix == NULL) {
        throw invalid_argument("elementalMatrix == NULL");
    }
    if (gamma == NULL) {
        throw invalid_argument("gamma == NULL");
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_matrix_kernel<<<blocks, threads, 0, stream>>>(
        matrix->values, matrix->columnIds, this->mSMatrix->columnIds,
        this->mConnectivityMatrix->deviceData, elementalMatrix->deviceData,
        gamma->deviceData, this->mSigmaRef, this->mConnectivityMatrix->rows,
        matrix->density);
}

// specialisation
template class Model<Basis>;
