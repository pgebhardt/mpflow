// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"
#include "../include/forward.hcu"

// calc jacobian kernel
template<
    int nodes_per_element
>
__global__ void calcJacobianKernel(const fastEIT::dtype::real* drivePhi,
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
        (measurmentCount + fastEIT::Matrix<fastEIT::dtype::real>::block_size - 1) /
        fastEIT::Matrix<fastEIT::dtype::real>::block_size) *
        fastEIT::Matrix<fastEIT::dtype::real>::block_size;
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
        jacobian[row + column * rows] += -element;
    }
    else {
        jacobian[row + column * rows] = -element;
    }
}

// calc jacobian
template <
    int nodes_per_element
>
void fastEIT::forward::calcJacobian(const Matrix<dtype::real>& gamma, const Matrix<dtype::real>& phi,
    const Matrix<dtype::index>& elements, const Matrix<dtype::real>& elemental_jacobian_matrix,
    dtype::size drive_count, dtype::size measurment_count, dtype::real sigma_ref, bool additiv,
    cudaStream_t stream, Matrix<dtype::real>* jacobian) {
    // check input
    if (jacobian == NULL) {
        throw std::invalid_argument("ForwardSolver::calcJacobian: jacobian == NULL");
    }

    // dimension
    dim3 blocks(jacobian->data_rows() / fastEIT::Matrix<fastEIT::dtype::real>::block_size,
        jacobian->data_columns() / fastEIT::Matrix<fastEIT::dtype::real>::block_size);
    dim3 threads(fastEIT::Matrix<fastEIT::dtype::real>::block_size,
        fastEIT::Matrix<fastEIT::dtype::real>::block_size);

    // calc jacobian
    calcJacobianKernel<nodes_per_element><<<blocks, threads, 0, stream>>>(
        phi.device_data(), &phi.device_data()[drive_count * phi.data_rows()],
        elements.device_data(), elemental_jacobian_matrix.device_data(),
        gamma.device_data(), sigma_ref, jacobian->data_rows(), jacobian->data_columns(),
        phi.data_rows(), elements.rows(), drive_count, measurment_count, additiv,
        jacobian->device_data());
}

// template specialisation
template void fastEIT::forward::calcJacobian<3>(
    const fastEIT::Matrix<fastEIT::dtype::real>&, const fastEIT::Matrix<fastEIT::dtype::real>&,
    const fastEIT::Matrix<fastEIT::dtype::index>&, const fastEIT::Matrix<fastEIT::dtype::real>&,
    fastEIT::dtype::size, fastEIT::dtype::size, fastEIT::dtype::real, bool,
    cudaStream_t, fastEIT::Matrix<fastEIT::dtype::real>*);
