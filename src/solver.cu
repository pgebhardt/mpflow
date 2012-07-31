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
#include "../include/fastect.h"

// calc jacobian kernel
__global__ void calc_jacobian_kernel(linalgcu_matrix_data_t* jacobian,
    linalgcu_matrix_data_t* applied_phi,
    linalgcu_matrix_data_t* lead_phi,
    linalgcu_matrix_data_t* gradient_matrix_values,
    linalgcu_column_id_t* gradient_matrix_column_ids,
    linalgcu_matrix_data_t* area, linalgcu_size_t jacobian_size_m,
    linalgcu_size_t phi_size_m, linalgcu_size_t measurment_count,
    linalgcu_size_t element_count) {
    // get id
    linalgcu_size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcu_size_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= element_count) {
        return;
    }

    // calc measurment and drive id
    linalgcu_size_t measurment_id = i % measurment_count;
    linalgcu_size_t drive_id = i / measurment_count;

    // memory
    linalgcu_matrix_data_t element = 0.0f;
    float2 grad_applied_phi = {0.0f, 0.0f};
    float2 grad_lead_phi = {0.0f, 0.0f};
    linalgcu_column_id_t idx, idy;

    // calc x components
    for (int k = 0; k < 3; k++) {
        idx = gradient_matrix_column_ids[2 * j * LINALGCU_BLOCK_SIZE + k];

        if (idx == -1) {
            break;
        }

        // x gradient
        grad_applied_phi.x += gradient_matrix_values[2 * j * LINALGCU_BLOCK_SIZE + k] *
            applied_phi[idx + drive_id * phi_size_m];
        grad_lead_phi.x += gradient_matrix_values[2 * j * LINALGCU_BLOCK_SIZE + k] *
            lead_phi[idx + measurment_id * phi_size_m];
    }

    // calc y components
    for (int k = 0; k < 3; k++) {
        idy = gradient_matrix_column_ids[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k];

        if (idy == -1) {
            break;
        }

        // y gradient
        grad_applied_phi.y += gradient_matrix_values[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k] *
            applied_phi[idy + drive_id * phi_size_m];
        grad_lead_phi.y += gradient_matrix_values[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k] *
            lead_phi[idy + measurment_id * phi_size_m];
    }

    // calc matrix element
    element = -area[j] * (grad_applied_phi.x * grad_lead_phi.x +
        grad_applied_phi.y * grad_lead_phi.y);

    // set matrix element
    jacobian[i + j * jacobian_size_m] = element;
}

// calc jacobian
extern "C"
linalgcu_error_t fastect_solver_calc_jacobian(fastect_solver_t solver,
    cudaStream_t stream) {
    if (solver == NULL) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(solver->jacobian->size_m / LINALGCU_BLOCK_SIZE,
        solver->jacobian->size_n / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // calc jacobian
    calc_jacobian_kernel<<<blocks, threads, 0, stream>>>(
        solver->jacobian->device_data,
        solver->applied_solver->phi->device_data,
        solver->lead_solver->phi->device_data,
        solver->applied_solver->grid->gradient_matrix_sparse->values,
        solver->applied_solver->grid->gradient_matrix_sparse->column_ids,
        solver->applied_solver->grid->area->device_data,
        solver->jacobian->size_m, solver->applied_solver->phi->size_m,
        solver->lead_solver->phi->size_n, solver->mesh->element_count);

    return LINALGCU_SUCCESS;
}
