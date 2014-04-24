// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/eit/model_kernel.h"

// reduce connectivity and elementalResidual matrix
template <
    class type
>
static __global__ void reduceMatrixKernel(const type* intermediate_matrix,
    const mpFlow::dtype::index* column_ids, mpFlow::dtype::size rows,
    mpFlow::dtype::index offset, type* matrix) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    mpFlow::dtype::index columnId = column_ids[row * mpFlow::numeric::sparseMatrix::block_size + column];

    // check column id
    if (columnId == mpFlow::dtype::invalid_index) {
        return;
    }

    // reduce matrices
    matrix[row + (column + offset * mpFlow::numeric::sparseMatrix::block_size) * rows] =
        intermediate_matrix[row + columnId * rows];
}

// reduce matrix wrapper
template <
    class type
>
void mpFlow::EIT::modelKernel::reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* intermediate_matrix, const dtype::index* column_ids, dtype::size rows,
    dtype::index offset, type* matrix) {
    // call cuda kernel
    reduceMatrixKernel<type><<<blocks, threads, 0, stream>>>(intermediate_matrix,
        column_ids, rows, offset, matrix);

    CudaCheckError();
}

// reduce matrix specialisation
template void mpFlow::EIT::modelKernel::reduceMatrix<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::real*);
template void mpFlow::EIT::modelKernel::reduceMatrix<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::index*);

// update matrix kernel
static __global__ void updateMatrixKernel(const mpFlow::dtype::index* connectivityMatrix,
    const mpFlow::dtype::real* elementalMatrix, const mpFlow::dtype::real* gamma,
    mpFlow::dtype::real sigmaRef, mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    mpFlow::dtype::real* matrix_values) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    mpFlow::dtype::real value = 0.0f;
    mpFlow::dtype::index elementId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index k = 0; k < columns / mpFlow::numeric::sparseMatrix::block_size; ++k) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows];

        value += elementId != mpFlow::dtype::invalid_index ? elementalMatrix[row +
            (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] = value;
}

// update matrix kernel wrapper
void mpFlow::EIT::modelKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
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
    const mpFlow::dtype::real* s_matrix_values, const mpFlow::dtype::real* r_matrix_values,
    const mpFlow::dtype::index* s_matrix_column_ids, const mpFlow::dtype::real* z_matrix,
    mpFlow::dtype::size density, mpFlow::dtype::real scalar, mpFlow::dtype::size z_matrix_rows,
    mpFlow::dtype::real* system_matrix_values) {
    // get row
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // update system matrix
    mpFlow::dtype::index column_id = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0; column < density; ++column) {
        // get column id
        column_id = s_matrix_column_ids[row * mpFlow::numeric::sparseMatrix::block_size + column];

        // update system matrix element
        system_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] =
            column_id != mpFlow::dtype::invalid_index ?
            s_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] +
            r_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] * scalar +
            z_matrix[row + z_matrix_rows * column_id] :
            system_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column];
    }
}

// update system matrix kernel wrapper
void mpFlow::EIT::modelKernel::updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const mpFlow::dtype::real* s_matrix_values, const mpFlow::dtype::real* r_matrix_values,
    const mpFlow::dtype::index* s_matrix_column_ids, const mpFlow::dtype::real* z_matrix,
    mpFlow::dtype::size density, mpFlow::dtype::real scalar, mpFlow::dtype::size z_matrix_rows,
    mpFlow::dtype::real* system_matrix_values) {
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
static __global__ void calcJacobianKernel(const mpFlow::dtype::real* drivePhi,
    const mpFlow::dtype::real* measurmentPhi, const mpFlow::dtype::index* connectivityMatrix,
    const mpFlow::dtype::real* elementalJacobianMatrix, const mpFlow::dtype::real* gamma,
    mpFlow::dtype::real sigmaRef, mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    mpFlow::dtype::size phiRows, mpFlow::dtype::size elementCount,
    mpFlow::dtype::size driveCount, mpFlow::dtype::size measurmentCount, bool additiv,
    mpFlow::dtype::real* jacobian) {
    // get id
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    mpFlow::dtype::size roundMeasurmentCount = (
        (measurmentCount + mpFlow::numeric::matrix::block_size - 1) /
        mpFlow::numeric::matrix::block_size) *
        mpFlow::numeric::matrix::block_size;
    mpFlow::dtype::size measurmentId = row % roundMeasurmentCount;
    mpFlow::dtype::size driveId = row / roundMeasurmentCount;

    // variables
    mpFlow::dtype::real dPhi[nodes_per_element], mPhi[nodes_per_element];
    mpFlow::dtype::index index;

    // get data
    for (mpFlow::dtype::index i = 0; i < nodes_per_element; i++) {
        index = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[index + driveId * phiRows] : 0.0f;
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[index +
            measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    mpFlow::dtype::real element = 0.0f;
    for (mpFlow::dtype::index i = 0; i < nodes_per_element; i++)
    for (mpFlow::dtype::index j = 0; j < nodes_per_element; j++) {
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
void mpFlow::EIT::modelKernel::calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
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
template void mpFlow::EIT::modelKernel::calcJacobian<3>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
template void mpFlow::EIT::modelKernel::calcJacobian<6>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
