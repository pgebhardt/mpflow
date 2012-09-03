// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// add scalar kernel
__global__ void add_scalar_kernel(linalgcu_matrix_data_t* vector,
    linalgcu_matrix_data_t* scalar, linalgcu_size_t vector_rows,
    linalgcu_size_t rows, linalgcu_size_t columns) {
    // get row
    linalgcu_size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // get column
    linalgcu_size_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vector_rows] += row < rows && column < columns ? scalar[column * vector_rows] : 0.0f;
}

// add scalar
extern "C"
linalgcu_error_t fastect_conjugate_sparse_add_scalar(linalgcu_matrix_t vector,
    linalgcu_matrix_t scalar, linalgcu_size_t rows, linalgcu_size_t columns, cudaStream_t stream) {
    // check input
    if ((vector == NULL) || (scalar == NULL)) {
        return LINALGCU_ERROR;
    }

    // kernel dimension
    dim3 global(vector->rows / LINALGCU_BLOCK_SIZE, vector->columns);
    dim3 local(LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    add_scalar_kernel<<<global, local, 0, stream>>>(vector->device_data, scalar->device_data,
        vector->rows, rows, columns);

    return LINALGCU_SUCCESS;
}

// update vector
__global__ void sparse_update_vector_kernel(linalgcu_matrix_data_t* result,
    linalgcu_matrix_data_t* x1, linalgcu_matrix_data_t sign,
    linalgcu_matrix_data_t* x2, linalgcu_matrix_data_t* r1, linalgcu_matrix_data_t* r2,
    linalgcu_size_t rows) {
    // get row
    linalgcu_size_t row = blockIdx.x * blockDim.x + threadIdx.x;

    // get column
    linalgcu_size_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != 0.0f ? x1[row + column * rows] + sign * x2[row + column * rows] *
        r1[column * rows] / r2[column * rows] : 0.0f;
}

// update vector
extern "C"
linalgcu_error_t fastect_conjugate_sparse_update_vector(linalgcu_matrix_t result,
    linalgcu_matrix_t x1, linalgcu_matrix_data_t sign, linalgcu_matrix_t x2,
    linalgcu_matrix_t r1, linalgcu_matrix_t r2, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (x1 == NULL) || (x2 == NULL) || (r1 == NULL) || (r2 == NULL)) {
        return LINALGCU_ERROR;
    }

    // kernel dimension
    dim3 global(result->rows / LINALGCU_BLOCK_SIZE, result->columns);
    dim3 local(LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    sparse_update_vector_kernel<<<global, local, 0, stream>>>(result->device_data, x1->device_data, sign, x2->device_data,
        r1->device_data, r2->device_data, result->rows);

    return LINALGCU_SUCCESS;
}
