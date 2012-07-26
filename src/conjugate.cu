// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cuda/cuda.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>
#include "conjugate.h"

// add scalar kernel
__global__ void add_scalar_kernel(linalgcu_matrix_data_t* vector,
    linalgcu_matrix_data_t* scalar) {
    // get id
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // add data
    vector[i] += scalar[0];
}

// add scalar
extern "C"
linalgcu_error_t ert_conjugate_add_scalar(linalgcu_matrix_t vector,
    linalgcu_matrix_t scalar, cudaStream_t stream) {
    // check input
    if ((vector == NULL) || (scalar == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    add_scalar_kernel<<<vector->size_m / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE,
        0, stream>>>(vector->device_data, scalar->device_data);

    return LINALGCU_SUCCESS;
}

// update vector
__global__ void update_vector_kernel(linalgcu_matrix_data_t* result,
    linalgcu_matrix_data_t* x1, linalgcu_matrix_data_t sign,
    linalgcu_matrix_data_t* x2, linalgcu_matrix_data_t* r1, linalgcu_matrix_data_t* r2) {
    // get id
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // calc value
    result[i] = x1[i] + sign * x2[i] * r1[0] / r2[0];
}

// update vector
extern "C"
linalgcu_error_t ert_conjugate_udate_vector(linalgcu_matrix_t result,
    linalgcu_matrix_t x1, linalgcu_matrix_data_t sign, linalgcu_matrix_t x2,
    linalgcu_matrix_t r1, linalgcu_matrix_t r2, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (x1 == NULL) || (x2 == NULL) || (r1 == NULL) || (r2 == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    update_vector_kernel<<<result->size_m / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE,
        0, stream>>>(result->device_data, x1->device_data, sign, x2->device_data,
        r1->device_data, r2->device_data);

    return LINALGCU_SUCCESS;
}

