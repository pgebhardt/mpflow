// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../include/cuda_error.h"

#include "../include/dtype.h"
#include "../include/constants.h"
#include "../include/forward_kernel.h"

// calc jacobian kernel
template <
    int nodes_per_element
>
static __global__ void calcJacobianKernel(const fastEIT::dtype::real* drivePhi,
    const fastEIT::dtype::real* measurmentPhi, const fastEIT::dtype::index* connectivityMatrix,
    const fastEIT::dtype::real* elementalJacobianMatrix, const fastEIT::dtype::real* gamma,
    fastEIT::dtype::real sigmaRef, fastEIT::dtype::size rows, fastEIT::dtype::size columns,
    fastEIT::dtype::size phiRows, fastEIT::dtype::size elementCount,
    fastEIT::dtype::size driveCount, fastEIT::dtype::size measurmentCount, bool additiv,
    fastEIT::dtype::real* jacobian) {
    // get id
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    fastEIT::dtype::size roundMeasurmentCount = (
        (measurmentCount + fastEIT::matrix::block_size - 1) /
        fastEIT::matrix::block_size) *
        fastEIT::matrix::block_size;
    fastEIT::dtype::size measurmentId = row % roundMeasurmentCount;
    fastEIT::dtype::size driveId = row / roundMeasurmentCount;

    // variables
    fastEIT::dtype::real dPhi[nodes_per_element], mPhi[nodes_per_element];
    fastEIT::dtype::index index;

    // get data
    for (fastEIT::dtype::index i = 0; i < nodes_per_element; i++) {
        index = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[index + driveId * phiRows] : 0.0f;
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[index +
            measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    fastEIT::dtype::real element = 0.0f;
    for (fastEIT::dtype::index i = 0; i < nodes_per_element; i++) {
        for (fastEIT::dtype::index j = 0; j < nodes_per_element; j++) {
            element += dPhi[i] * mPhi[j] * elementalJacobianMatrix[column +
                (i + j * nodes_per_element) * columns];
        }
    }

    // diff sigma to gamma
    element *= sigmaRef * exp10f(gamma[column] / 10.0f) / 10.0f;

    // set matrix element
    if (additiv == true) {
        jacobian[row + column * rows] += element;
    }
    else {
        jacobian[row + column * rows] = element;
    }
}

// calc jacobian kernel wrapper
template <
    int nodes_per_element
>
void fastEIT::forwardKernel::calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::real* drive_phi, const dtype::real* measurment_phi,
    const dtype::index* connectivity_matrix, const dtype::real* elemental_jacobian_matrix,
    const dtype::real* gamma, dtype::real sigma_ref, dtype::size rows, dtype::size columns,
    dtype::size phi_rows, dtype::size element_count, dtype::size drive_count,
    dtype::size measurment_count, bool additiv, dtype::real* jacobian) {
    // call cuda kernel
    calcJacobianKernel<nodes_per_element><<<blocks, threads, 0, stream>>>(
        drive_phi, measurment_phi, connectivity_matrix, elemental_jacobian_matrix, gamma,
        sigma_ref, rows, columns, phi_rows, element_count, drive_count,
        measurment_count, additiv, jacobian);

    CudaCheckError();
}

// calc voltage kernel
static __global__ void calcVoltageKernel(const fastEIT::dtype::real* potential,
    fastEIT::dtype::size offset, fastEIT::dtype::size rows, const fastEIT::dtype::real* pattern,
    fastEIT::dtype::size pattern_rows, bool additiv,
    fastEIT::dtype::real* voltage, fastEIT::dtype::size voltage_rows) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc voltage
    fastEIT::dtype::real value = 0.0f;
    for (fastEIT::dtype::index electrode = 0; electrode < pattern_rows; ++electrode) {
        value += pattern[electrode + pattern_rows * row] * potential[offset + electrode + column * rows];
    }

    // set voltage
    if (additiv == true) {
        voltage[row + voltage_rows * column] += value;
    } else {
        voltage[row + voltage_rows * column] = value;
    }
}

// calc voltage kernel wrapper
void fastEIT::forwardKernel::calcVoltage(dim3 blocks, dim3 threads, cudaStream_t stream,
    const fastEIT::dtype::real* potential, fastEIT::dtype::size offset,
    fastEIT::dtype::size rows, const fastEIT::dtype::real* pattern,
    fastEIT::dtype::size pattern_rows, bool additiv,
    fastEIT::dtype::real* voltage, fastEIT::dtype::size voltage_rows) {
    // call cuda kernel
    calcVoltageKernel<<<blocks, threads, 0, stream>>>(
        potential, offset, rows, pattern, pattern_rows, additiv, voltage, voltage_rows);

    CudaCheckError();
}


// template specialisation
template void fastEIT::forwardKernel::calcJacobian<3>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
