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
#include "basis.h"
#include "mesh.h"
#include "electrodes.h"
#include "grid.h"

// update_system_matrix_kernel
__global__ void update_system_matrix_kernel(linalgcu_matrix_data_t* system_matrix_values,
    linalgcu_column_id_t* system_matrix_column_ids,
    linalgcu_matrix_data_t* gradient_matrix_transposed_values,
    linalgcu_column_id_t* gradient_matrix_transposed_column_ids,
    linalgcu_matrix_data_t* gradient_matrix_transposed,
    linalgcu_matrix_data_t* sigma,
    linalgcu_matrix_data_t* area,
    linalgcu_size_t gradient_matrix_transposed_size_m) {
    // get ids
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcu_column_id_t j = system_matrix_column_ids[i * LINALGCU_BLOCK_SIZE +
        (blockIdx.y * blockDim.y + threadIdx.y)];

    // calc system matrix elements
    linalgcu_matrix_data_t element = 0.0f;
    linalgcu_column_id_t id = -1;

    for (linalgcu_size_t k = 0; k < LINALGCU_BLOCK_SIZE; k++) {
        // get id
        id = gradient_matrix_transposed_column_ids[i * LINALGCU_BLOCK_SIZE + k];

        element += id != -1 && j != -1 ? gradient_matrix_transposed_values[i * LINALGCU_BLOCK_SIZE + k] *
            sigma[id / 2] * area[id / 2] *
            gradient_matrix_transposed[j + id * gradient_matrix_transposed_size_m] :
            0.0f;
    }

    // set element
    system_matrix_values[i * LINALGCU_BLOCK_SIZE + (blockIdx.y * blockDim.y + threadIdx.y)] =
        element;
}

// update system matrix
extern "C"
linalgcu_error_t ert_grid_update_system_matrix(ert_grid_t grid, cudaStream_t stream) {
    // check input
    if (grid == NULL) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(grid->system_matrix->size_m / LINALGCU_BLOCK_SIZE, 1);

    // execute kernel
    update_system_matrix_kernel<<<blocks, threads, 0, stream>>>(
        grid->system_matrix->values,
        grid->system_matrix->column_ids,
        grid->gradient_matrix_transposed_sparse->values,
        grid->gradient_matrix_transposed_sparse->column_ids,
        grid->gradient_matrix_transposed->device_data,
        grid->sigma->device_data,
        grid->area->device_data,
        grid->gradient_matrix_transposed->size_m);

    return LINALGCU_SUCCESS;
}

