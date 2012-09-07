// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

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
    linalgcu_matrix_data_t* area, linalgcu_size_t jacobian_rows,
    linalgcu_size_t phi_rows, linalgcu_size_t measurment_count,
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
            applied_phi[idx + drive_id * phi_rows];
        grad_lead_phi.x += gradient_matrix_values[2 * j * LINALGCU_BLOCK_SIZE + k] *
            lead_phi[idx + measurment_id * phi_rows];
    }

    // calc y components
    for (int k = 0; k < 3; k++) {
        idy = gradient_matrix_column_ids[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k];

        if (idy == -1) {
            break;
        }

        // y gradient
        grad_applied_phi.y += gradient_matrix_values[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k] *
            applied_phi[idy + drive_id * phi_rows];
        grad_lead_phi.y += gradient_matrix_values[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k] *
            lead_phi[idy + measurment_id * phi_rows];
    }

    // calc matrix element
    element = -area[j] * (grad_applied_phi.x * grad_lead_phi.x +
        grad_applied_phi.y * grad_lead_phi.y);

    // set matrix element
    jacobian[i + j * jacobian_rows] = element;
}

// calc jacobian
extern "C"
linalgcu_error_t fastect_forward_solver_calc_jacobian(fastect_forward_solver_t solver,
    linalgcu_matrix_t jacobian, linalgcu_size_t harmonic, cudaStream_t stream) {
    if ((solver == NULL) || (jacobian == NULL) || (harmonic > solver->grid->numHarmonics)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(jacobian->rows / LINALGCU_BLOCK_SIZE, jacobian->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // calc jacobian
    calc_jacobian_kernel<<<blocks, threads, 0, stream>>>(
        jacobian->deviceData,
        solver->drivePhi[harmonic]->deviceData,
        solver->measurmentPhi[harmonic]->deviceData,
        solver->grid->gradientMatrixSparse->values,
        solver->grid->gradientMatrixSparse->columnIds,
        solver->grid->area->deviceData,
        jacobian->rows, solver->drivePhi[harmonic]->rows,
        solver->measurmentPhi[harmonic]->columns, solver->grid->mesh->elementCount);

    return LINALGCU_SUCCESS;
}
