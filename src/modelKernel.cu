// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"
#include "../include/sparse.hpp"
#include "../include/modelKernel.hpp"

// reduce connectivity and elementalResidual matrix
template <class type>
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
void fastEIT::modelKernel::reduceMatrix(const Matrix<dtype::real>& intermediateMatrix, const SparseMatrix& shape, cudaStream_t stream,
    Matrix<dtype::real>& matrix) {
    // block size
    dim3 blocks(matrix.data_rows() / Matrix<dtype::real>::block_size, 1);
    dim3 threads(Matrix<dtype::real>::block_size, Matrix<dtype::real>::block_size);

    // reduce matrix
    reduceMatrixKernel<dtype::real><<<blocks, threads, 0, stream>>>(
        intermediateMatrix.device_data(), shape.column_ids(), matrix.data_rows(),
        shape.density(), matrix.set_device_data());
}
void fastEIT::modelKernel::reduceMatrix(const Matrix<dtype::index>& intermediateMatrix, const SparseMatrix& shape, cudaStream_t stream,
    Matrix<dtype::index>& matrix) {
    // block size
    dim3 blocks(matrix.data_rows() / Matrix<dtype::real>::block_size, 1);
    dim3 threads(Matrix<dtype::real>::block_size, Matrix<dtype::real>::block_size);

    // reduce matrix
    reduceMatrixKernel<dtype::index><<<blocks, threads, 0, stream>>>(
        intermediateMatrix.device_data(), shape.column_ids(), matrix.data_rows(),
        shape.density(), matrix.set_device_data());
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
void fastEIT::modelKernel::updateMatrix(const Matrix<dtype::real>& elements, const Matrix<dtype::real>& gamma,
    const Matrix<dtype::index>& connectivityMatrix, dtype::real sigmaRef, cudaStream_t stream, SparseMatrix& matrix) {
    // dimension
    dim3 threads(Matrix<dtype::real>::block_size, Matrix<dtype::real>::block_size);
    dim3 blocks(matrix.data_rows() / Matrix<dtype::real>::block_size, 1);

    // execute kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(
        connectivityMatrix.device_data(), elements.device_data(), gamma.device_data(),
        sigmaRef, connectivityMatrix.data_rows(), matrix.density(), matrix.set_values());
}
