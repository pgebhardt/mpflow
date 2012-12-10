// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/sparse_matrix.h"
#include "../include/model_cuda.h"

// reduce connectivity and elementalResidual matrix
template <
    class type
>
__global__ void reduceMatrixKernel(const type* intermediateMatrix, const fastEIT::dtype::index* systemMatrixColumnIds,
    fastEIT::dtype::size rows, fastEIT::dtype::size density, type* matrix) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    fastEIT::dtype::index columnId = systemMatrixColumnIds[row * fastEIT::SparseMatrix::block_size + column];

    // check column id
    if (columnId == -1) {
        return;
    }

    // reduce matrices
    for (fastEIT::dtype::index k = 0; k < density; ++k) {
        matrix[row + (column + k * fastEIT::SparseMatrix::block_size) * rows] =
            intermediateMatrix[row + (columnId + k * rows) * rows];
    }
}

// reduce matrix
template <
    class type
>
void fastEIT::model::reduceMatrix(const Matrix<type>* intermediateMatrix,
    const SparseMatrix* shape, cudaStream_t stream, Matrix<type>* matrix) {
    // check input
    if (intermediateMatrix == NULL) {
        throw std::invalid_argument("model::reduceMatrix: intermediateMatrix == NULL");
    }
    if (shape == NULL) {
        throw std::invalid_argument("model::reduceMatrix: shape == NULL");
    }
    if (matrix == NULL) {
        throw std::invalid_argument("model::reduceMatrix: matrix == NULL");
    }

    // block size
    dim3 blocks(matrix->data_rows() / Matrix<dtype::real>::block_size, 1);
    dim3 threads(Matrix<dtype::real>::block_size, Matrix<dtype::real>::block_size);

    // reduce matrix
    reduceMatrixKernel<type><<<blocks, threads, 0, stream>>>(
        intermediateMatrix->device_data(), shape->column_ids(), matrix->data_rows(),
        shape->density(), matrix->device_data());
}

// update matrix kernel
__global__ void updateMatrixKernel(const fastEIT::dtype::index* connectivityMatrix,
    const fastEIT::dtype::real* elementalMatrix, const fastEIT::dtype::real* gamma,
    fastEIT::dtype::real sigmaRef, fastEIT::dtype::size rows,
    fastEIT::dtype::size density, fastEIT::dtype::real* matrixValues) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    fastEIT::dtype::real value = 0.0f;
    fastEIT::dtype::index elementId = -1;
    for (fastEIT::dtype::index k = 0; k < density; ++k) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * fastEIT::SparseMatrix::block_size) * rows];

        value += elementId != -1 ? elementalMatrix[row +
            (column + k * fastEIT::SparseMatrix::block_size) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrixValues[row * fastEIT::SparseMatrix::block_size + column] = value;
}

// update matrix
void fastEIT::model::updateMatrix(const Matrix<dtype::real>* elements,
    const Matrix<dtype::real>* gamma, const Matrix<dtype::index>* connectivityMatrix,
    dtype::real sigmaRef, cudaStream_t stream, SparseMatrix* matrix) {
    // check input
    if (elements == NULL) {
        throw std::invalid_argument("model::updateMatrix: elements == NULL");
    }
    if (gamma == NULL) {
        throw std::invalid_argument("model::updateMatrix: gamma == NULL");
    }
    if (connectivityMatrix == NULL) {
        throw std::invalid_argument("model::updateMatrix: connectivityMatrix == NULL");
    }
    if (matrix == NULL) {
        throw std::invalid_argument("model::updateMatrix: matrix == NULL");
    }

    // dimension
    dim3 threads(Matrix<dtype::real>::block_size, Matrix<dtype::real>::block_size);
    dim3 blocks(matrix->data_rows() / Matrix<dtype::real>::block_size, 1);

    // execute kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(
        connectivityMatrix->device_data(), elements->device_data(), gamma->device_data(),
        sigmaRef, connectivityMatrix->data_rows(), matrix->density(), matrix->values());
}

// template specialisation
template void fastEIT::model::reduceMatrix<fastEIT::dtype::real>(const Matrix<fastEIT::dtype::real>*, const SparseMatrix*,
    cudaStream_t, Matrix<fastEIT::dtype::real>*);
template void fastEIT::model::reduceMatrix<fastEIT::dtype::index>(const Matrix<fastEIT::dtype::index>*, const SparseMatrix*,
    cudaStream_t, Matrix<fastEIT::dtype::index>*);
