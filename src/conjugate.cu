// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace fastEIT::numeric;
using namespace std;

// add scalar kernel
__global__ void addScalarKernel(dtype::real* vector, dtype::real* scalar,
    dtype::size vectorRows, dtype::size rows, dtype::size columns) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vectorRows] += row < rows && column < columns ? scalar[column * vectorRows] : 0.0f;
}

// add scalar
void Conjugate::addScalar(Matrix<dtype::real>* vector,
    Matrix<dtype::real>* scalar, dtype::size rows, dtype::size columns, cudaStream_t stream) {
    // check input
    if (vector == NULL) {
        throw invalid_argument("Conjugate::addScalar: vector == NULL");
    }
    if (scalar == NULL) {
        throw invalid_argument("Conjugate::addScalar: scalar == NULL");
    }

    // kernel dimension
    dim3 global(vector->dataRows() / Matrix<dtype::real>::blockSize, vector->dataColumns() == 1 ? 1 :
        vector->dataColumns() / Matrix<dtype::real>::blockSize);
    dim3 local(Matrix<dtype::real>::blockSize, vector->dataColumns() == 1 ? 1 : Matrix<dtype::real>::blockSize);

    // execute kernel
    addScalarKernel<<<global, local, 0, stream>>>(vector->deviceData(), scalar->deviceData(),
        vector->dataRows(), rows, columns);
}

// update vector
__global__ void updateVectorKernel(dtype::real* result, dtype::real* x1, dtype::real sign,
    dtype::real* x2, dtype::real* r1, dtype::real* r2, dtype::size rows) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] + sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector
void Conjugate::updateVector(Matrix<dtype::real>* result,
    Matrix<dtype::real>* x1, dtype::real sign, Matrix<dtype::real>* x2,
    Matrix<dtype::real>* r1, Matrix<dtype::real>* r2, cudaStream_t stream) {
    if (result == NULL) {
        throw invalid_argument("Conjugate::addScalar: result == NULL");
    }
    if (x1 == NULL) {
        throw invalid_argument("Conjugate::addScalar: x1 == NULL");
    }
    if (x2 == NULL) {
        throw invalid_argument("Conjugate::addScalar: x2 == NULL");
    }
    if (r1 == NULL) {
        throw invalid_argument("Conjugate::addScalar: r1 == NULL");
    }
    if (r2 == NULL) {
        throw invalid_argument("Conjugate::addScalar: r2 == NULL");
    }

    // kernel dimension
    dim3 global(result->dataRows() / Matrix<dtype::real>::blockSize, result->dataColumns() == 1 ? 1 :
        result->dataColumns() / Matrix<dtype::real>::blockSize);
    dim3 local(Matrix<dtype::real>::blockSize, result->dataColumns() == 1 ? 1 : Matrix<dtype::real>::blockSize);

    // execute kernel
    updateVectorKernel<<<global, local, 0, stream>>>(result->deviceData(),
        x1->deviceData(), sign, x2->deviceData(), r1->deviceData(), r2->deviceData(), result->dataRows());
}

// gemv kernel
__global__ void gemvKernel(dtype::real* matrix, dtype::real* vector,
    dtype::real* result, dtype::size rows) {
    // get ids
    dtype::index row = threadIdx.x + blockIdx.x * blockDim.x;
    dtype::index column = (threadIdx.y + blockIdx.y * blockDim.y) * 2 * Matrix<dtype::real>::blockSize;

    // load vector to shared memory
    __shared__ dtype::real work[2 * Matrix<dtype::real>::blockSize * Matrix<dtype::real>::blockSize];
    work[threadIdx.x + threadIdx.y * 2 * Matrix<dtype::real>::blockSize] = column + threadIdx.x < rows ?
        vector[column + threadIdx.x] : 0.0f;
    __syncthreads();

    // compute partial vector product
    dtype::real product = 0.0f;
    for (dtype::index i = 0; i < 2 * Matrix<dtype::real>::blockSize; i++) {
        product += row < rows && column + i < rows ? matrix[row + (column + i) * rows] * work[i + threadIdx.y * 2 * Matrix<dtype::real>::blockSize] : 0.0f;
    }

    // set result
    if (row < rows) {
        result[row + (threadIdx.y + blockIdx.y * blockDim.y) * rows] = product;
    }
}

// row reduce kernel
__global__ void reduceRowKernel(dtype::real* vector, dtype::size rows) {
    // get id
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // check row
    if (row >= rows) {
        return;
    }

    // sum row
    dtype::real sum = 0.0f;
    dtype::size count = (rows + 2 * Matrix<dtype::real>::blockSize - 1) / (2 * Matrix<dtype::real>::blockSize);
    for (dtype::index i = 0; i < count; i++) {
        sum += vector[row + i * rows];
    }

    // set sum
    vector[row] = sum;
}

// fast gemv
void Conjugate::gemv(Matrix<dtype::real>* result, Matrix<dtype::real>* matrix,
    Matrix<dtype::real>* vector, cudaStream_t stream) {
    // check input
    if (result == NULL) {
        throw invalid_argument("Conjugate::addScalar: result == NULL");
    }
    if (matrix == NULL) {
        throw invalid_argument("Conjugate::addScalar: matrix == NULL");
    }
    if (vector == NULL) {
        throw invalid_argument("Conjugate::addScalar: vector == NULL");
    }

    // dimension
    dim3 blocks((matrix->dataRows() + 2 * Matrix<dtype::real>::blockSize - 1) / (2 * Matrix<dtype::real>::blockSize),
        (matrix->dataRows() / (2 * Matrix<dtype::real>::blockSize) + Matrix<dtype::real>::blockSize - 1) / Matrix<dtype::real>::blockSize);
    dim3 threads(2 * Matrix<dtype::real>::blockSize, Matrix<dtype::real>::blockSize);

    // call gemv kernel
    gemvKernel<<<blocks, threads, 0, stream>>>(matrix->deviceData(), vector->deviceData(),
        result->deviceData(), matrix->dataRows());

    // call reduce kernel
    reduceRowKernel<<<(matrix->dataColumns() + Matrix<dtype::real>::blockSize * Matrix<dtype::real>::blockSize - 1) /
        (Matrix<dtype::real>::blockSize * Matrix<dtype::real>::blockSize),
        Matrix<dtype::real>::blockSize * Matrix<dtype::real>::blockSize, 0, stream>>>(result->deviceData(), result->dataRows());
}

