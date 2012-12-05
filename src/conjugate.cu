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
#include "../include/conjugate_cuda.h"

// add scalar kernel
__global__ void addScalarKernel(const fastEIT::dtype::real* scalar,
    fastEIT::dtype::size vectorRows, fastEIT::dtype::size rows,
    fastEIT::dtype::size columns, fastEIT::dtype::real* vector) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vectorRows] += row < rows && column < columns ?
        scalar[column * vectorRows] : 0.0f;
}

// add scalar
void fastEIT::numeric::conjugate::addScalar(const Matrix<dtype::real>& scalar,
    dtype::size rows, dtype::size columns, cudaStream_t stream, Matrix<dtype::real>* vector) {
    // check input
    if (vector == NULL) {
        throw std::invalid_argument("Conjugate::addScalar: vector == NULL");
    }

    // kernel dimension
    dim3 global(vector->data_rows() / Matrix<dtype::real>::block_size,
        vector->data_columns() == 1 ? 1 :
        vector->data_columns() / Matrix<dtype::real>::block_size);
    dim3 local(Matrix<dtype::real>::block_size,
        vector->data_columns() == 1 ? 1 : Matrix<dtype::real>::block_size);

    // execute kernel
    addScalarKernel<<<global, local, 0, stream>>>(scalar.device_data(),
        vector->data_rows(), rows, columns, vector->device_data());
}

// update vector
__global__ void updateVectorKernel(const fastEIT::dtype::real* x1,
    const fastEIT::dtype::real sign, const fastEIT::dtype::real* x2,
    const fastEIT::dtype::real* r1, const fastEIT::dtype::real* r2,
    fastEIT::dtype::size rows, fastEIT::dtype::real* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] +
        sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector
void fastEIT::numeric::conjugate::updateVector(const Matrix<dtype::real>& x1,
    dtype::real sign, const Matrix<dtype::real>& x2, const Matrix<dtype::real>& r1,
    const Matrix<dtype::real>& r2, cudaStream_t stream, Matrix<dtype::real>* result) {
    if (result == NULL) {
        throw std::invalid_argument("Conjugate::addScalar: result == NULL");
    }

    // kernel dimension
    dim3 global(result->data_rows() / Matrix<dtype::real>::block_size,
        result->data_columns() == 1 ? 1 :
        result->data_columns() / Matrix<dtype::real>::block_size);
    dim3 local(Matrix<dtype::real>::block_size,
        result->data_columns() == 1 ? 1 : Matrix<dtype::real>::block_size);

    // execute kernel
    updateVectorKernel<<<global, local, 0, stream>>>(x1.device_data(), sign,
        x2.device_data(), r1.device_data(), r2.device_data(), result->data_rows(),
        result->device_data());
}

// gemv kernel
__global__ void gemvKernel(const fastEIT::dtype::real* matrix,
    const fastEIT::dtype::real* vector, fastEIT::dtype::size rows,
    fastEIT::dtype::real* result) {
    // get ids
    fastEIT::dtype::index row = threadIdx.x + blockIdx.x * blockDim.x;
    fastEIT::dtype::index column = (threadIdx.y + blockIdx.y * blockDim.y) *
        2 * fastEIT::Matrix<fastEIT::dtype::real>::block_size;

    // load vector to shared memory
    __shared__ fastEIT::dtype::real work[2 *
        fastEIT::Matrix<fastEIT::dtype::real>::block_size *
        fastEIT::Matrix<fastEIT::dtype::real>::block_size];
    work[threadIdx.x +
        threadIdx.y * 2 * fastEIT::Matrix<fastEIT::dtype::real>::block_size] =
        column + threadIdx.x < rows ? vector[column + threadIdx.x] : 0.0f;
    __syncthreads();

    // compute partial vector product
    fastEIT::dtype::real product = 0.0f;
    for (fastEIT::dtype::index i = 0;
        i < 2 * fastEIT::Matrix<fastEIT::dtype::real>::block_size;
        i++) {
        product += row < rows && column + i < rows ?
            matrix[row + (column + i) * rows] * work[i +
            threadIdx.y * 2 * fastEIT::Matrix<fastEIT::dtype::real>::block_size] :
            0.0f;
    }

    // set result
    if (row < rows) {
        result[row + (threadIdx.y + blockIdx.y * blockDim.y) * rows] = product;
    }
}

// row reduce kernel
__global__ void reduceRowKernel(fastEIT::dtype::size rows,
    fastEIT::dtype::real* vector) {
    // get id
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // check row
    if (row >= rows) {
        return;
    }

    // sum row
    fastEIT::dtype::real sum = 0.0f;
    fastEIT::dtype::size count =
        (rows + 2 * fastEIT::Matrix<fastEIT::dtype::real>::block_size - 1) /
        (2 * fastEIT::Matrix<fastEIT::dtype::real>::block_size);
    for (fastEIT::dtype::index i = 0; i < count; i++) {
        sum += vector[row + i * rows];
    }

    // set sum
    vector[row] = sum;
}

// fast gemv
void fastEIT::numeric::conjugate::gemv(const Matrix<dtype::real>& matrix,
    const Matrix<dtype::real>& vector, cudaStream_t stream, Matrix<dtype::real>* result) {
    // check input
    if (result == NULL) {
        throw std::invalid_argument("Conjugate::addScalar: result == NULL");
    }

    // dimension
    dim3 blocks(
        (matrix.data_rows() + 2 * Matrix<dtype::real>::block_size - 1) /
        (2 * Matrix<dtype::real>::block_size),
        (matrix.data_rows() / (2 * Matrix<dtype::real>::block_size) +
        Matrix<dtype::real>::block_size - 1) / Matrix<dtype::real>::block_size);
    dim3 threads(2 * Matrix<dtype::real>::block_size, Matrix<dtype::real>::block_size);

    // call gemv kernel
    gemvKernel<<<blocks, threads, 0, stream>>>(matrix.device_data(), vector.device_data(),
        matrix.data_rows(), result->device_data());

    // call reduce kernel
    reduceRowKernel<<<(matrix.data_columns() +
        Matrix<dtype::real>::block_size * Matrix<dtype::real>::block_size - 1) /
        (Matrix<dtype::real>::block_size * Matrix<dtype::real>::block_size),
        Matrix<dtype::real>::block_size * Matrix<dtype::real>::block_size, 0, stream>>>(
            result->data_rows(), result->device_data());
}

