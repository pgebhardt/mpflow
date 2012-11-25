// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// reduce connectivity and elementalResidual matrix
template <class type>
__global__ void reduceMatrixKernel(type* matrix,
    type* intermediateMatrix, dtype::index* systemMatrixColumnIds,
    dtype::size rows, dtype::size density) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    dtype::index columnId = systemMatrixColumnIds[row * SparseMatrix::blockSize + column];

    // check column id
    if (columnId == -1) {
        return;
    }

    // reduce matrices
    for (dtype::index k = 0; k < density; k++) {
        matrix[row + (column + k * SparseMatrix::blockSize) * rows] =
            intermediateMatrix[row + (columnId + k * rows) * rows];
    }
}

// reduce matrix
template <class BasisFunction>
void Model<BasisFunction>::reduceMatrix(Matrix<dtype::real>* matrix,
    Matrix<dtype::real>* intermediateMatrix, dtype::size density, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("Model::reduceMatrix: matrix == NULL");
    }
    if (intermediateMatrix == NULL) {
        throw invalid_argument("Model::reduceMatrix: intermediateMatrix == NULL");
    }

    // block size
    dim3 blocks(matrix->dataRows() / Matrix<dtype::real>::blockSize, 1);
    dim3 threads(Matrix<dtype::real>::blockSize, Matrix<dtype::real>::blockSize);

    // reduce matrix
    reduceMatrixKernel<dtype::real><<<blocks, threads, 0, stream>>>(
        matrix->deviceData(), intermediateMatrix->deviceData(),
        this->SMatrix()->columnIds(), matrix->dataRows(),
        density);
}
template <class BasisFunction>
void Model<BasisFunction>::reduceMatrix(Matrix<dtype::index>* matrix,
    Matrix<dtype::index>* intermediateMatrix, dtype::size density, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("Model::reduceMatrix: matrix == NULL");
    }
    if (intermediateMatrix == NULL) {
        throw invalid_argument("Model::reduceMatrix: intermediateMatrix == NULL");
    }

    // block size
    dim3 blocks(matrix->dataRows() / Matrix<dtype::index>::blockSize, 1);
    dim3 threads(Matrix<dtype::index>::blockSize, Matrix<dtype::index>::blockSize);

    // reduce matrix
    reduceMatrixKernel<dtype::index><<<blocks, threads, 0, stream>>>(
        matrix->deviceData(), intermediateMatrix->deviceData(),
        this->SMatrix()->columnIds(), matrix->dataRows(),
        density);
}

// update matrix kernel
__global__ void updateMatrixKernel(dtype::real* matrixValues,
    dtype::index* connectivityMatrix, dtype::real* elementalMatrix,
    dtype::real* gamma, dtype::real sigmaRef, dtype::size rows,
    dtype::size density) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    dtype::real value = 0.0f;
    dtype::index elementId = -1;
    for (dtype::index k = 0; k < density; k++) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * SparseMatrix::blockSize) * rows];

        value += elementId != -1 ? elementalMatrix[row +
            (column + k * SparseMatrix::blockSize) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrixValues[row * SparseMatrix::blockSize + column] = value;
}

// update matrix
template <class BasisFunction>
void Model<BasisFunction>::updateMatrix(SparseMatrix* matrix,
    Matrix<dtype::real>* elementalMatrix, Matrix<dtype::real>* gamma, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("Model::updateMatrix: matrix == NULL");
    }
    if (elementalMatrix == NULL) {
        throw invalid_argument("Model::updateMatrix: elementalMatrix == NULL");
    }
    if (gamma == NULL) {
        throw invalid_argument("Model::updateMatrix: gamma == NULL");
    }

    // dimension
    dim3 threads(Matrix<dtype::real>::blockSize, Matrix<dtype::real>::blockSize);
    dim3 blocks(matrix->dataRows() / Matrix<dtype::real>::blockSize, 1);

    // execute kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(
        matrix->values(), this->connectivityMatrix()->deviceData(),
        elementalMatrix->deviceData(), gamma->deviceData(), this->sigmaRef(),
        this->connectivityMatrix()->dataRows(), matrix->density());
}

// specialisation
template class fastEIT::Model<fastEIT::Basis::Linear>;
