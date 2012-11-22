// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// add scalar kernel
__global__ void add_scalar_kernel(dtype::real* vector,
    dtype::real* scalar, linalgcuSize_t vector_rows,
    linalgcuSize_t rows, linalgcuSize_t columns) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vector_rows] += row < rows && column < columns ? scalar[column * vector_rows] : 0.0f;
}

// add scalar
void Conjugate::add_scalar(linalgcuMatrix_t vector,
    linalgcuMatrix_t scalar, linalgcuSize_t rows, linalgcuSize_t columns, cudaStream_t stream) {
    // check input
    if (vector == NULL) {
        throw invalid_argument("Conjugate::add_scalar: vector == NULL");
    }
    if (scalar == NULL) {
        throw invalid_argument("Conjugate::add_scalar: scalar == NULL");
    }

    // kernel dimension
    dim3 global(vector->rows / LINALGCU_BLOCK_SIZE, vector->columns == 1 ? 1 :
        vector->columns / LINALGCU_BLOCK_SIZE);
    dim3 local(LINALGCU_BLOCK_SIZE, vector->columns == 1 ? 1 : LINALGCU_BLOCK_SIZE);

    // execute kernel
    add_scalar_kernel<<<global, local, 0, stream>>>(vector->deviceData, scalar->deviceData,
        vector->rows, rows, columns);
}

// update vector
__global__ void update_vector_kernel(dtype::real* result,
    dtype::real* x1, dtype::real sign,
    dtype::real* x2, dtype::real* r1, dtype::real* r2,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] + sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector
void Conjugate::update_vector(linalgcuMatrix_t result,
    linalgcuMatrix_t x1, dtype::real sign, linalgcuMatrix_t x2,
    linalgcuMatrix_t r1, linalgcuMatrix_t r2, cudaStream_t stream) {
    // check input
    if (result == NULL) {
        throw invalid_argument("Conjugate::update_vector: result == NULL");
    }
    if (x1 == NULL) {
        throw invalid_argument("Conjugate::update_vector: x1 == NULL");
    }
    if (x2 == NULL) {
        throw invalid_argument("Conjugate::update_vector: x2 == NULL");
    }
    if (r1 == NULL) {
        throw invalid_argument("Conjugate::update_vector: r1 == NULL");
    }
    if (r2 == NULL) {
        throw invalid_argument("Conjugate::update_vector: r2 == NULL");
    }

    // kernel dimension
    dim3 global(result->rows / LINALGCU_BLOCK_SIZE, result->columns == 1 ? 1 :
        result->columns / LINALGCU_BLOCK_SIZE);
    dim3 local(LINALGCU_BLOCK_SIZE, result->columns == 1 ? 1 : LINALGCU_BLOCK_SIZE);

    // execute kernel
    update_vector_kernel<<<global, local, 0, stream>>>(result->deviceData,
        x1->deviceData, sign, x2->deviceData, r1->deviceData, r2->deviceData, result->rows);
}

// gemv kernel
__global__ void gemv_kernel(dtype::real* matrix, dtype::real* vector,
    dtype::real* result, linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = threadIdx.x + blockIdx.x * blockDim.x;
    linalgcuSize_t column = (threadIdx.y + blockIdx.y * blockDim.y) * 2 * LINALGCU_BLOCK_SIZE;

    // load vector to shared memory
    __shared__ dtype::real work[2 * LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE];
    work[threadIdx.x + threadIdx.y * 2 * LINALGCU_BLOCK_SIZE] = column + threadIdx.x < rows ?
        vector[column + threadIdx.x] : 0.0f;
    __syncthreads();

    // compute partial vector product
    dtype::real product = 0.0f;
    for (int i = 0; i < 2 * LINALGCU_BLOCK_SIZE; i++) {
        product += row < rows && column + i < rows ? matrix[row + (column + i) * rows] * work[i + threadIdx.y * 2 * LINALGCU_BLOCK_SIZE] : 0.0f;
    }

    // set result
    if (row < rows) {
        result[row + (threadIdx.y + blockIdx.y * blockDim.y) * rows] = product;
    }
}

// row reduce kernel
__global__ void reduce_row_kernel(dtype::real* vector, linalgcuSize_t rows) {
    // get id
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // check row
    if (row >= rows) {
        return;
    }

    // sum row
    dtype::real sum = 0.0f;
    linalgcuSize_t count = (rows + 2 * LINALGCU_BLOCK_SIZE - 1) / (2 * LINALGCU_BLOCK_SIZE);
    for (int i = 0; i < count; i++) {
        sum += vector[row + i * rows];
    }

    // set sum
    vector[row] = sum;
}

// fast gemv
void Conjugate::gemv(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
    linalgcuMatrix_t vector, cudaStream_t stream) {
    // check input
    if (result == NULL) {
        throw invalid_argument("Conjugate::gemv: result == NULL");
    }
    if (matrix == NULL) {
        throw invalid_argument("Conjugate::gemv: matrix == NULL");
    }
    if (vector == NULL) {
        throw invalid_argument("Conjugate::gemv: vector == NULL");
    }

    // dimension
    dim3 blocks((matrix->rows + 2 * LINALGCU_BLOCK_SIZE - 1) / (2 * LINALGCU_BLOCK_SIZE),
        (matrix->rows / (2 * LINALGCU_BLOCK_SIZE) + LINALGCU_BLOCK_SIZE - 1) / LINALGCU_BLOCK_SIZE);
    dim3 threads(2 * LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // call gemv kernel
    gemv_kernel<<<blocks, threads, 0, stream>>>(matrix->deviceData, vector->deviceData,
        result->deviceData, matrix->rows);

    // call reduce kernel
    reduce_row_kernel<<<(matrix->columns + LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE - 1) /
        (LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE), LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE,
        0, stream>>>(result->deviceData, result->rows);
}

