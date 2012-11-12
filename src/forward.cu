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
    linalgcuMatrixData_t* connectivityMatrix,
    linalgcuMatrixData_t* elementalJacobianMatrix,
    linalgcuMatrixData_t* gamma, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t rows, linalgcuSize_t columns,
    linalgcuSize_t phiRows, linalgcuSize_t elementCount,
    linalgcuSize_t driveCount, linalgcuSize_t measurmentCount, linalgcuBool_t additiv) {
    // get id
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    linalgcuSize_t roundMeasurmentCount = ((measurmentCount + LINALGCU_BLOCK_SIZE - 1) /
        LINALGCU_BLOCK_SIZE) * LINALGCU_BLOCK_SIZE;
    linalgcuSize_t measurmentId = row % roundMeasurmentCount;
    linalgcuSize_t driveId = row / roundMeasurmentCount;

    // variables
    linalgcuMatrixData_t dPhi[FASTEIT_NODES_PER_ELEMENT], mPhi[FASTEIT_NODES_PER_ELEMENT];
    linalgcuMatrixData_t id;

    // get data
    for (int i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
        id = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[(linalgcuSize_t)id + driveId * phiRows] : 0.0f;
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[(linalgcuSize_t)id +
            measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    linalgcuMatrixData_t element = 0.0f;
    for (int i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
        for (int j = 0; j < FASTEIT_NODES_PER_ELEMENT; j++) {
            element += dPhi[i] * mPhi[j] * elementalJacobianMatrix[column +
                (i + j * FASTEIT_NODES_PER_ELEMENT) * columns];
        }
    }

    // diff sigma to gamma
    element *= sigmaRef * exp10f(gamma[column] / 10.0f) / 10.0f;

    // set matrix element
    if (additiv == LINALGCU_TRUE) {
        jacobian[row + column * rows] += -element;
    }
    else {
        jacobian[row + column * rows] = -element;
    }
}

// calc jacobian
LINALGCU_EXTERN_C
linalgcuError_t fasteit_forward_solver_calc_jacobian(fasteitForwardSolver_t self,
    linalgcuMatrix_t gamma, linalgcuSize_t harmonic, linalgcuBool_t additiv,
    cudaStream_t stream) {
    if ((self == NULL) || (gamma == NULL) || (harmonic > self->model->numHarmonics)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(self->jacobian->rows / LINALGCU_BLOCK_SIZE,
        self->jacobian->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // calc jacobian
    calc_jacobian_kernel<<<blocks, threads, 0, stream>>>(
        self->jacobian->deviceData, self->phi[harmonic]->deviceData,
        &self->phi[harmonic]->deviceData[self->driveCount * self->phi[harmonic]->rows],
        self->model->mesh->elements->deviceData, self->elementalJacobianMatrix->deviceData,
        gamma->deviceData, self->model->sigmaRef, self->jacobian->rows, self->jacobian->columns,
        self->phi[harmonic]->rows, self->model->mesh->elementCount,
        self->driveCount, self->measurmentCount, additiv);

    return LINALGCU_SUCCESS;
}
