// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/fasteit.h"

// calc jacobian kernel
__global__ void calc_jacobian_kernel(linalgcuMatrixData_t* jacobian,
    linalgcuMatrixData_t* drivePhi,
    linalgcuMatrixData_t* measurmentPhi,
    linalgcuMatrixData_t* gradientMatrixValues,
    linalgcuColumnId_t* gradientMatrixColumnIds,
    linalgcuMatrixData_t* area, linalgcuMatrixData_t* gamma,
    linalgcuMatrixData_t sigmaRef, linalgcuSize_t jacobianRows,
    linalgcuSize_t phiRows, linalgcuSize_t driveCount,
    linalgcuSize_t measurmentCount, linalgcuSize_t elementCount,
    linalgcuBool_t additiv) {
    // get id
    linalgcuSize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t j = blockIdx.y * blockDim.y + threadIdx.y;

    if (j >= elementCount) {
        return;
    }

    // calc measurment and drive id
    linalgcuSize_t measurmentId = i % 16;
    linalgcuSize_t driveId = i / 16;

    // memory
    linalgcuMatrixData_t element = 0.0f;
    float2 gradDrivePhi = {0.0f, 0.0f};
    float2 gradMeasurmentPhi = {0.0f, 0.0f};
    linalgcuColumnId_t idx, idy;

    // calc x components
    linalgcuMatrixData_t value = 0.0f;
    for (int k = 0; k < 3; k++) {
        idx = gradientMatrixColumnIds[2 * j * LINALGCU_BLOCK_SIZE + k];

        if (idx == -1) {
            break;
        }

        // read x gradient value
        value = gradientMatrixValues[2 * j * LINALGCU_BLOCK_SIZE + k];

        // x gradient
        gradDrivePhi.x += driveId < driveCount ? value * drivePhi[idx + driveId * phiRows] : 0.0f;
        gradMeasurmentPhi.x += measurmentId < measurmentCount ? value * measurmentPhi[idx + measurmentId * phiRows] : 0.0f;
    }

    // calc y components
    for (int k = 0; k < 3; k++) {
        idy = gradientMatrixColumnIds[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k];

        if (idy == -1) {
            break;
        }

        // read y gradient value
        value = gradientMatrixValues[(2 * j + 1) * LINALGCU_BLOCK_SIZE + k];

        // y gradient
        gradDrivePhi.y += driveId < driveCount ? value * drivePhi[idy + driveId * phiRows] : 0.0f;
        gradMeasurmentPhi.y += measurmentId < measurmentCount ? value * measurmentPhi[idy + measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    element = -area[j] * (gradDrivePhi.x * gradMeasurmentPhi.x +
        gradDrivePhi.y * gradMeasurmentPhi.y) * (sigmaRef * exp10f(gamma[j] / 10.0f) / 10.0f);

    // set matrix element
    if (additiv == LINALGCU_TRUE) {
        jacobian[i + j * jacobianRows] += element;
    }
    else {
        jacobian[i + j * jacobianRows] = element;
    }
}

// calc jacobian
LINALGCU_EXTERN_C
linalgcuError_t fasteit_forward_solver_calc_jacobian(fasteitForwardSolver_t self,
    linalgcuMatrix_t gamma, linalgcuSize_t harmonic, linalgcuBool_t additiv,
    cudaStream_t stream) {
    if ((self == NULL) || (gamma == NULL) || (harmonic > self->grid->numHarmonics)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(self->jacobian->rows / LINALGCU_BLOCK_SIZE,
        self->jacobian->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // calc jacobian
    calc_jacobian_kernel<<<blocks, threads, 0, stream>>>(
        self->jacobian->deviceData,
        self->phi[harmonic]->deviceData,
        &self->phi[harmonic]->deviceData[self->driveCount * self->phi[harmonic]->rows],
        self->grid->gradientMatrixSparse->values,
        self->grid->gradientMatrixSparse->columnIds,
        self->grid->area->deviceData, gamma->deviceData, self->grid->sigmaRef,
        self->jacobian->rows, self->phi[harmonic]->rows,
        self->driveCount, self->measurmentCount, self->grid->mesh->elementCount,
        additiv);

    return LINALGCU_SUCCESS;
}
