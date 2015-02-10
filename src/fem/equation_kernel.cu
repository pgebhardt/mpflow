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
#include <thrust/complex.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/fem/equation_kernel.h"

// reduce connectivity and elementalResidual matrix
template <
    class inputType,
    class outputType
>
static __global__ void reduceMatrixKernel(const inputType* intermediate_matrix,
    const mpFlow::dtype::index* column_ids, mpFlow::dtype::size rows,
    mpFlow::dtype::index offset, outputType* matrix) {
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
    class inputType,
    class outputType
>
void mpFlow::FEM::equationKernel::reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const inputType* intermediate_matrix, const dtype::index* column_ids, dtype::size rows,
    dtype::index offset, outputType* matrix) {
    // call cuda kernel
    reduceMatrixKernel<<<blocks, threads, 0, stream>>>(intermediate_matrix,
        column_ids, rows, offset, matrix);

    CudaCheckError();
}

// reduce matrix specialisation
template void mpFlow::FEM::equationKernel::reduceMatrix<mpFlow::dtype::real, mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::real*);
template void mpFlow::FEM::equationKernel::reduceMatrix<mpFlow::dtype::real, mpFlow::dtype::complex>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::complex*);
template void mpFlow::FEM::equationKernel::reduceMatrix<mpFlow::dtype::index, mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::index*);

// update matrix kernel
template <
    class dataType
>
static __global__ void updateMatrixKernel(const mpFlow::dtype::index* connectivityMatrix,
    const dataType* elementalMatrix, const dataType* gamma,
    dataType referenceValue, mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    dataType* matrix_values) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    dataType value = 0.0f;
    mpFlow::dtype::index elementId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index k = 0; k < columns / mpFlow::numeric::sparseMatrix::block_size; ++k) {
        // get element id
        elementId = connectivityMatrix[row + (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows];

        value += elementId != mpFlow::dtype::invalid_index ?
            elementalMatrix[row + (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows] *
            referenceValue * exp(log((mpFlow::dtype::real)10.0) * gamma[elementId] / (mpFlow::dtype::real)10.0) :
            (mpFlow::dtype::real)0.0;
    }

    // set residual matrix element
    matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] = value;
}

// update matrix kernel wrapper
template <
    class dataType
>
void mpFlow::FEM::equationKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::index* connectivityMatrix, const dataType* elementalMatrix,
    const dataType* gamma, dataType referenceValue, dtype::size rows, dtype::size columns,
    dataType* matrix_values) {
    // call cuda kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(connectivityMatrix, elementalMatrix,
        gamma, referenceValue, rows, columns, matrix_values);

    CudaCheckError();
}

template void mpFlow::FEM::equationKernel::updateMatrix<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::real*,
    const mpFlow::dtype::real*, mpFlow::dtype::real, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::FEM::equationKernel::updateMatrix<mpFlow::dtype::complex>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::complex*,
    const mpFlow::dtype::complex*, mpFlow::dtype::complex, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::complex*);

// update system matrix kernel
template <
    class dataType
>
static __global__ void updateSystemMatrixKernel(
    const dataType* sMatrixValues, const dataType* rMatrixValues,
    const mpFlow::dtype::index* sMatrixColumnIds, mpFlow::dtype::size density,
    dataType k, dataType* systemMatrixValues) {
    // get row
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // update system matrix
    mpFlow::dtype::index columnId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0; column < density; ++column) {
        // get column id
        columnId = sMatrixColumnIds[row * mpFlow::numeric::sparseMatrix::block_size + column];

        // update system matrix element
        systemMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column] =
            columnId != mpFlow::dtype::invalid_index ?
            sMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column] +
            rMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column] * k :
            systemMatrixValues[row * mpFlow::numeric::sparseMatrix::block_size + column];
    }
}

// update system matrix kernel wrapper
template <
    class dataType
>
void mpFlow::FEM::equationKernel::updateSystemMatrix(
    dim3 blocks, dim3 threads, cudaStream_t stream,
    const dataType* sMatrixValues, const dataType* rMatrixValues,
    const dtype::index* sMatrixColumnIds, dtype::size density, dataType k,
    dataType* systemMatrixValues) {
    // call cuda kernel
    updateSystemMatrixKernel<<<blocks, threads, 0, stream>>>(
        sMatrixValues, rMatrixValues, sMatrixColumnIds, density, k,
        systemMatrixValues);

    CudaCheckError();
}

template void mpFlow::FEM::equationKernel::updateSystemMatrix<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::real*,
    const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::real,
    mpFlow::dtype::real*);
template void mpFlow::FEM::equationKernel::updateSystemMatrix<mpFlow::dtype::complex>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::complex*, const mpFlow::dtype::complex*,
    const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::complex,
    mpFlow::dtype::complex*);

// calc jacobian kernel
template <
    class dataType,
    int nodes_per_element
>
static __global__ void calcJacobianKernel(const dataType* drivePhi,
    const dataType* measurmentPhi, const mpFlow::dtype::index* connectivityMatrix,
    const mpFlow::dtype::real* elementalJacobianMatrix, const dataType* factor,
    dataType referenceValue, mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    mpFlow::dtype::size phiRows, mpFlow::dtype::size elementCount,
    mpFlow::dtype::size driveCount, mpFlow::dtype::size measurmentCount, bool additiv,
    dataType* jacobian) {
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
    dataType dPhi[nodes_per_element], mPhi[nodes_per_element];
    mpFlow::dtype::index index;

    // get data
    for (mpFlow::dtype::index i = 0; i < nodes_per_element; i++) {
        index = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[index + driveId * phiRows] : 0.0f;
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[index +
            measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    dataType element = 0.0f;
    for (mpFlow::dtype::index i = 0; i < nodes_per_element; i++)
    for (mpFlow::dtype::index j = 0; j < nodes_per_element; j++) {
        element += dPhi[i] * mPhi[j] * elementalJacobianMatrix[column +
            (i + j * nodes_per_element) * columns];
    }

    // diff sigma to gamma
    element *= referenceValue * exp(log((mpFlow::dtype::real)10.0) * factor[column] / (mpFlow::dtype::real)10.0) / (mpFlow::dtype::real)10.0;

    if (additiv) {
        jacobian[row + column * rows] += element;
    }
    else {
        jacobian[row + column * rows] = element;
    }
}

// calc jacobian kernel wrapper
template <
    class dataType,
    int nodes_per_element
>
void mpFlow::FEM::equationKernel::calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dataType* drive_phi, const dataType* measurment_phi,
    const dtype::index* connectivity_matrix, const dtype::real* elemental_jacobian_matrix,
    const dataType* factor, dataType referenceValue, dtype::size rows, dtype::size columns,
    dtype::size phi_rows, dtype::size element_count, dtype::size drive_count,
    dtype::size measurment_count, bool additiv, dataType* jacobian) {
    // call cuda kernel
    calcJacobianKernel<dataType, nodes_per_element><<<blocks, threads, 0, stream>>>(
        drive_phi, measurment_phi, connectivity_matrix, elemental_jacobian_matrix, factor,
        referenceValue, rows, columns, phi_rows, element_count, drive_count,
        measurment_count, additiv, jacobian);

    CudaCheckError();
}

// template specialisation
template void mpFlow::FEM::equationKernel::calcJacobian<mpFlow::dtype::real, 3>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
template void mpFlow::FEM::equationKernel::calcJacobian<mpFlow::dtype::real, 6>(dim3, dim3, cudaStream_t,
    const dtype::real*, const dtype::real*, const dtype::index*, const dtype::real*,
    const dtype::real*, dtype::real, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::real*);
template void mpFlow::FEM::equationKernel::calcJacobian<mpFlow::dtype::complex, 3>(dim3, dim3, cudaStream_t,
    const dtype::complex*, const dtype::complex*, const dtype::index*, const dtype::real*,
    const dtype::complex*, dtype::complex, dtype::size, dtype::size, dtype::size, dtype::size,
    dtype::size, dtype::size, bool, dtype::complex*);
