// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fastect.h"

// calc jacobian kernel
__global__ void calc_jacobian_kernel(linalgcuMatrixData_t* jacobian,
    linalgcuMatrixData_t* applied_phi,
    linalgcuMatrixData_t* lead_phi,
    linalgcuMatrixData_t* gradient_matrix_values,
    linalgcuColumnId_t* gradient_matrix_column_ids,
    linalgcuMatrixData_t* area, linalgcuMatrixData_t* gamma,
    linalgcuMatrixData_t sigmaRef, linalgcuSize_t jacobian_rows,
    linalgcuSize_t phi_rows, linalgcuSize_t measurment_count,
    linalgcuSize_t element_count, linalgcuBool_t additiv) {
    // get id
    linalgcuSize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= element_count) {
        return;
    }

    // calc measurment and drive id
    linalgcuSize_t measurment_id = i % measurment_count;
    linalgcuSize_t drive_id = i / measurment_count;

    // memory
    linalgcuMatrixData_t element = 0.0f;
    float2 grad_applied_phi = {0.0f, 0.0f};
    float2 grad_lead_phi = {0.0f, 0.0f};
    linalgcuColumnId_t idx, idy;

    // calc x components
    linalgcuMatrixData_t value = 0.0f;
    for (int k = 0; k < 3; k++) {
        idx = gradient_matrix_column_ids[2 * j * LINALGCU_BLOCK_SIZE + k];

        if (idx == -1) {
            break;
        }

        // read x gradient value
        value = gradient_matrix_values[2 * j * LINALGCU_BLOCK_SIZE + k];

        // x gradient
        grad_applied_phi.x += value * applied_phi[idx + drive_id * phi_rows];
        grad_lead_phi.x += value * lead_phi[idx + measurment_id * phi_rows];
    }

    // calc y components
    for (int k = 0; k < 3; k++) {
        idy = gradient_matrix_column_ids[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k];

        if (idy == -1) {
            break;
        }

        // read y gradient value
        value = gradient_matrix_values[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k];

        // y gradient
        grad_applied_phi.y += value * applied_phi[idy + drive_id * phi_rows];
        grad_lead_phi.y += value * lead_phi[idy + measurment_id * phi_rows];
    }

    // calc matrix element
    element = -area[j] * (grad_applied_phi.x * grad_lead_phi.x +
        grad_applied_phi.y * grad_lead_phi.y) * (sigmaRef * exp10f(gamma[j] / 10.0f) / 10.0f);

    // set matrix element
    if (additiv == LINALGCU_TRUE) {
        jacobian[i + j * jacobian_rows] += element;
    }
    else {
        jacobian[i + j * jacobian_rows] = element;
    }
}

// calc jacobian
LINALGCU_EXTERN_C
linalgcuError_t fastect_forward_solver_calc_jacobian(fastectForwardSolver_t solver,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t gamma, linalgcuSize_t harmonic,
    linalgcuBool_t additiv, cudaStream_t stream) {
    if ((solver == NULL) || (jacobian == NULL) || (gamma == NULL) ||
        (harmonic > solver->grid->numHarmonics)) {
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
        solver->grid->area->deviceData, gamma->deviceData, solver->grid->sigmaRef,
        jacobian->rows, solver->drivePhi[harmonic]->rows,
        solver->measurmentPhi[harmonic]->columns, solver->grid->mesh->elementCount,
        additiv);

    return LINALGCU_SUCCESS;
}
