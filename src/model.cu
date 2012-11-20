// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

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
        throw invalid_argument("Model::reduce_matrix: matrix == NULL");
    }
    if (intermediateMatrix == NULL) {
        throw invalid_argument("Model::reduce_matrix: intermediateMatrix == NULL");
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
        throw invalid_argument("Model::update_matrix: matrix == NULL");
    }
    if (elementalMatrix == NULL) {
        throw invalid_argument("Model::update_matrix: elementalMatrix == NULL");
    }
    if (gamma == NULL) {
        throw invalid_argument("Model::update_matrix: gamma == NULL");
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
template class Model<LinearBasis>;
