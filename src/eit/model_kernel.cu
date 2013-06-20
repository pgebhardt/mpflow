// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fasteit/cuda_error.h"

#include "fasteit/dtype.h"
#include "fasteit/constants.h"
#include "fasteit/model_kernel.h"

// reduce connectivity and elementalResidual matrix
template <
    class type
>
static __global__ void reduceMatrixKernel(const type* intermediate_matrix,
    const fastEIT::dtype::index* column_ids, fastEIT::dtype::size rows,
    fastEIT::dtype::index offset, type* matrix) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    fastEIT::dtype::index columnId = column_ids[row * fastEIT::sparseMatrix::block_size + column];

    // check column id
    if (columnId == fastEIT::dtype::invalid_index) {
        return;
    }

    // reduce matrices
    matrix[row + (column + offset * fastEIT::sparseMatrix::block_size) * rows] =
        intermediate_matrix[row + columnId * rows];
}

// reduce matrix wrapper
template <
    class type
>
void fastEIT::modelKernel::reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* intermediate_matrix, const dtype::index* column_ids, dtype::size rows,
    dtype::index offset, type* matrix) {
    // call cuda kernel
    reduceMatrixKernel<type><<<blocks, threads, 0, stream>>>(intermediate_matrix,
        column_ids, rows, offset, matrix);

    CudaCheckError();
}

// reduce matrix specialisation
template void fastEIT::modelKernel::reduceMatrix<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::index, fastEIT::dtype::real*);
template void fastEIT::modelKernel::reduceMatrix<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::index, fastEIT::dtype::index*);

// update matrix kernel
static __global__ void updateMatrixKernel(const fastEIT::dtype::index* connectivityMatrix,
    const fastEIT::dtype::real* elementalMatrix, const fastEIT::dtype::real* gamma,
    fastEIT::dtype::real sigmaRef, fastEIT::dtype::size rows, fastEIT::dtype::size columns,
    fastEIT::dtype::real* matrix_values) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    fastEIT::dtype::real value = 0.0f;
    fastEIT::dtype::index elementId = fastEIT::dtype::invalid_index;
    for (fastEIT::dtype::index k = 0; k < columns / fastEIT::sparseMatrix::block_size; ++k) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * fastEIT::sparseMatrix::block_size) * rows];

        value += elementId != fastEIT::dtype::invalid_index ? elementalMatrix[row +
            (column + k * fastEIT::sparseMatrix::block_size) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrix_values[row * fastEIT::sparseMatrix::block_size + column] = value;
}

// update matrix kernel wrapper
void fastEIT::modelKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::index* connectivityMatrix, const dtype::real* elementalMatrix,
    const dtype::real* gamma, dtype::real sigma_ref, dtype::size rows, dtype::size columns,
    dtype::real* matrix_values) {
    // call cuda kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(connectivityMatrix, elementalMatrix,
        gamma, sigma_ref, rows, columns, matrix_values);

    CudaCheckError();
}

// update system matrix kernel
static __global__ void updateSystemMatrixKernel(
    const fastEIT::dtype::real* s_matrix_values, const fastEIT::dtype::real* r_matrix_values,
    const fastEIT::dtype::index* s_matrix_column_ids, const fastEIT::dtype::real* z_matrix,
    fastEIT::dtype::size density, fastEIT::dtype::real scalar, fastEIT::dtype::size z_matrix_rows,
    fastEIT::dtype::real* system_matrix_values) {
    // get row
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // update system matrix
    fastEIT::dtype::index column_id = fastEIT::dtype::invalid_index;
    for (fastEIT::dtype::index column = 0; column < density; ++column) {
        // get column id
        column_id = s_matrix_column_ids[row * fastEIT::sparseMatrix::block_size + column];

        // update system matrix element
        system_matrix_values[row * fastEIT::sparseMatrix::block_size + column] =
            column_id != fastEIT::dtype::invalid_index ?
            s_matrix_values[row * fastEIT::sparseMatrix::block_size + column] +
            r_matrix_values[row * fastEIT::sparseMatrix::block_size + column] * scalar +
            z_matrix[row + z_matrix_rows * column_id] :
            system_matrix_values[row * fastEIT::sparseMatrix::block_size + column];
    }
}

// update system matrix kernel wrapper
void fastEIT::modelKernel::updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const fastEIT::dtype::real* s_matrix_values, const fastEIT::dtype::real* r_matrix_values,
    const fastEIT::dtype::index* s_matrix_column_ids, const fastEIT::dtype::real* z_matrix,
    fastEIT::dtype::size density, fastEIT::dtype::real scalar, fastEIT::dtype::size z_matrix_rows,
    fastEIT::dtype::real* system_matrix_values) {
    // call cuda kernel
    updateSystemMatrixKernel<<<blocks, threads, 0, stream>>>(
        s_matrix_values, r_matrix_values, s_matrix_column_ids, z_matrix,
        density, scalar, z_matrix_rows, system_matrix_values);

    CudaCheckError();
}

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
    for (fastEIT::dtype::index i = 0; i < nodes_per_element; i++)
    for (fastEIT::dtype::index j = 0; j < nodes_per_element; j++) {
        element += dPhi[i] * mPhi[j] * elementalJacobianMatrix[column +
            (i + j * nodes_per_element) * columns];
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
void fastEIT::modelKernel::calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
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

// template specialisation
template void fastEIT::modelKernel::calcJacobian<3>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
template void fastEIT::modelKernel::calcJacobian<6>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
