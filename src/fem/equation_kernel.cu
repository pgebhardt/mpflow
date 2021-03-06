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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include "mpflow/cuda_error.h"

#include "mpflow/constants.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/type_traits.h"
#include "mpflow/fem/equation_kernel.h"

// logarithmic parametrization of material parameter
template <typename type, bool logarithmic>
inline __device__ type matParam(type const ref, type const v) {
    if (logarithmic == true) {
        return ref * exp(log(type(10)) * v / type(10));
    }
    else {
        return ref * v;
    }
}

template <typename type, bool logarithmic>
inline __device__ type matParamDeriv(type const ref, type const v) {
    if (logarithmic == true) {
        return (log(type(10)) / type(10)) * ref * exp(log(type(10)) * v / type(10));
    }
    else {
        return ref;
    }
}

// update matrix kernel
template <
    class dataType,
    bool logarithmic
>
static __global__ void updateMatrixKernel(const unsigned* connectivityMatrix,
    const dataType* elementalMatrix, const dataType* gamma,
    dataType referenceValue, unsigned rows, unsigned columns,
    dataType* matrix_values) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    dataType value = dataType(0);
    unsigned elementId = mpFlow::constants::invalidIndex;
    for (unsigned k = 0; k < columns / mpFlow::numeric::sparseMatrix::blockSize; ++k) {
        // get element id
        elementId = connectivityMatrix[row + (column + k * mpFlow::numeric::sparseMatrix::blockSize) * rows];

        value += elementId != mpFlow::constants::invalidIndex ?
            elementalMatrix[row + (column + k * mpFlow::numeric::sparseMatrix::blockSize) * rows] *
            matParam<dataType, logarithmic>(referenceValue, gamma[elementId]) : dataType(0);
    }

    // set residual matrix element
    matrix_values[row * mpFlow::numeric::sparseMatrix::blockSize + column] = value;
}

// update matrix kernel wrapper
template <
    class dataType,
    bool logarithmic
>
void mpFlow::FEM::equationKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const unsigned* connectivityMatrix, const dataType* elementalMatrix,
    const dataType* gamma, dataType referenceValue, unsigned rows, unsigned columns,
    dataType* matrix_values) {
    // call cuda kernel
    updateMatrixKernel<dataType, logarithmic><<<blocks, threads, 0, stream>>>(connectivityMatrix, elementalMatrix,
        gamma, referenceValue, rows, columns, matrix_values);

    CudaCheckError();
}

template void mpFlow::FEM::equationKernel::updateMatrix<float, true>(dim3, dim3,
    cudaStream_t, const unsigned*, const float*,
    const float*, float, unsigned,
    unsigned, float*);
template void mpFlow::FEM::equationKernel::updateMatrix<float, false>(dim3, dim3,
    cudaStream_t, const unsigned*, const float*,
    const float*, float, unsigned,
    unsigned, float*);
template void mpFlow::FEM::equationKernel::updateMatrix<double, true>(dim3, dim3,
    cudaStream_t, const unsigned*, const double*,
    const double*, double, unsigned,
    unsigned, double*);
template void mpFlow::FEM::equationKernel::updateMatrix<double, false>(dim3, dim3,
    cudaStream_t, const unsigned*, const double*,
    const double*, double, unsigned,
    unsigned, double*);
template void mpFlow::FEM::equationKernel::updateMatrix<thrust::complex<float>, true>(dim3, dim3,
    cudaStream_t, const unsigned*, const thrust::complex<float>*,
    const thrust::complex<float>*, thrust::complex<float>, unsigned,
    unsigned, thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::updateMatrix<thrust::complex<float>, false>(dim3, dim3,
    cudaStream_t, const unsigned*, const thrust::complex<float>*,
    const thrust::complex<float>*, thrust::complex<float>, unsigned,
    unsigned, thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::updateMatrix<thrust::complex<double>, true>(dim3, dim3,
    cudaStream_t, const unsigned*, const thrust::complex<double>*,
    const thrust::complex<double>*, thrust::complex<double>, unsigned,
    unsigned, thrust::complex<double>*);
template void mpFlow::FEM::equationKernel::updateMatrix<thrust::complex<double>, false>(dim3, dim3,
    cudaStream_t, const unsigned*, const thrust::complex<double>*,
    const thrust::complex<double>*, thrust::complex<double>, unsigned,
    unsigned, thrust::complex<double>*);

// update system matrix kernel
template <
    class dataType
>
static __global__ void updateSystemMatrixKernel(
    const dataType* sMatrixValues, const dataType* rMatrixValues,
    const unsigned* sMatrixColumnIds, unsigned density,
    dataType k, dataType* systemMatrixValues) {
    // get row
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;

    // update system matrix
    unsigned columnId = mpFlow::constants::invalidIndex;
    for (unsigned column = 0; column < density; ++column) {
        // get column id
        columnId = sMatrixColumnIds[row * mpFlow::numeric::sparseMatrix::blockSize + column];

        // update system matrix element
        systemMatrixValues[row * mpFlow::numeric::sparseMatrix::blockSize + column] =
            columnId != mpFlow::constants::invalidIndex ?
            sMatrixValues[row * mpFlow::numeric::sparseMatrix::blockSize + column] +
            rMatrixValues[row * mpFlow::numeric::sparseMatrix::blockSize + column] * k :
            systemMatrixValues[row * mpFlow::numeric::sparseMatrix::blockSize + column];
    }
}

// update system matrix kernel wrapper
template <
    class dataType
>
void mpFlow::FEM::equationKernel::updateSystemMatrix(
    dim3 blocks, dim3 threads, cudaStream_t stream,
    const dataType* sMatrixValues, const dataType* rMatrixValues,
    const unsigned* sMatrixColumnIds, unsigned density, dataType k,
    dataType* systemMatrixValues) {
    // call cuda kernel
    updateSystemMatrixKernel<<<blocks, threads, 0, stream>>>(
        sMatrixValues, rMatrixValues, sMatrixColumnIds, density, k,
        systemMatrixValues);

    CudaCheckError();
}

template void mpFlow::FEM::equationKernel::updateSystemMatrix<float>(dim3, dim3,
    cudaStream_t, const float*, const float*,
    const unsigned*, unsigned, float, float*);
template void mpFlow::FEM::equationKernel::updateSystemMatrix<double>(dim3, dim3,
    cudaStream_t, const double*, const double*,
    const unsigned*, unsigned, double, double*);
template void mpFlow::FEM::equationKernel::updateSystemMatrix<thrust::complex<float> >(dim3, dim3,
    cudaStream_t, const thrust::complex<float>*, const thrust::complex<float>*,
    const unsigned*, unsigned, thrust::complex<float>,
    thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::updateSystemMatrix<thrust::complex<double> >(dim3, dim3,
    cudaStream_t, const thrust::complex<double>*, const thrust::complex<double>*,
    const unsigned*, unsigned, thrust::complex<double>,
    thrust::complex<double>*);

// calc jacobian kernel
template <
    class dataType,
    int nodes_per_element,
    bool logarithmic
>
static __global__ void calcJacobianKernel(const dataType* drivePhi,
    const dataType* measurmentPhi, const unsigned* connectivityMatrix,
    const dataType* elementalJacobianMatrix, const dataType* gamma,
    dataType referenceValue, unsigned rows, unsigned columns,
    unsigned phiRows, unsigned elementCount,
    unsigned driveCount, unsigned measurmentCount, bool additiv,
    dataType* jacobian) {
    // get id
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned const column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    unsigned const roundMeasurmentCount = (
        (measurmentCount + mpFlow::numeric::matrix::blockSize - 1) /
        mpFlow::numeric::matrix::blockSize) *
        mpFlow::numeric::matrix::blockSize;
    unsigned const measurmentId = row % roundMeasurmentCount;
    unsigned const driveId = row / roundMeasurmentCount;

    // get data
    dataType dPhi[nodes_per_element], mPhi[nodes_per_element];
    for (unsigned i = 0; i < nodes_per_element; i++) {
        unsigned const index = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[index + driveId * phiRows] : dataType(0);
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[index +
            measurmentId * phiRows] : dataType(0);
    }

    // calc matrix element
    dataType element = dataType(0);
    for (unsigned i = 0; i < nodes_per_element; i++)
    for (unsigned j = 0; j < nodes_per_element; j++) {
        element += dPhi[i] * mPhi[j] * elementalJacobianMatrix[column +
            (i + j * nodes_per_element) * columns];
    }
    element *= matParamDeriv<dataType, logarithmic>(referenceValue, gamma[column]);

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
    int nodes_per_element,
    bool logarithmic
>
void mpFlow::FEM::equationKernel::calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dataType* drivePhi, const dataType* measurmentPhi,
    const unsigned* connectivityMatrix, const dataType* elementalJacobianMatrix,
    const dataType* gamma, dataType referenceValue, unsigned rows, unsigned columns,
    unsigned phiRows, unsigned elementCount, unsigned driveCount,
    unsigned measurmentCount, bool additiv, dataType* jacobian) {
    // call cuda kernel
    calcJacobianKernel<dataType, nodes_per_element, logarithmic><<<blocks, threads, 0, stream>>>(
        drivePhi, measurmentPhi, connectivityMatrix, elementalJacobianMatrix, gamma,
        referenceValue, rows, columns, phiRows, elementCount, driveCount,
        measurmentCount, additiv, jacobian);

    CudaCheckError();
}

// template specialisation
template void mpFlow::FEM::equationKernel::calcJacobian<float, 3, true>(dim3, dim3, cudaStream_t,
    const float*, const float*, const unsigned*, const float*,
    const float*, float, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, float*);
template void mpFlow::FEM::equationKernel::calcJacobian<float, 3, false>(dim3, dim3, cudaStream_t,
    const float*, const float*, const unsigned*, const float*,
    const float*, float, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, float*);
template void mpFlow::FEM::equationKernel::calcJacobian<float, 6, true>(dim3, dim3, cudaStream_t,
    const float*, const float*, const unsigned*, const float*,
    const float*, float, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, float*);
template void mpFlow::FEM::equationKernel::calcJacobian<float, 6, false>(dim3, dim3, cudaStream_t,
    const float*, const float*, const unsigned*, const float*,
    const float*, float, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, float*);
template void mpFlow::FEM::equationKernel::calcJacobian<double, 3, true>(dim3, dim3, cudaStream_t,
    const double*, const double*, const unsigned*, const double*,
    const double*, double, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, double*);
template void mpFlow::FEM::equationKernel::calcJacobian<double, 3, false>(dim3, dim3, cudaStream_t,
    const double*, const double*, const unsigned*, const double*,
    const double*, double, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, double*);
template void mpFlow::FEM::equationKernel::calcJacobian<double, 6, true>(dim3, dim3, cudaStream_t,
    const double*, const double*, const unsigned*, const double*,
    const double*, double, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, double*);
template void mpFlow::FEM::equationKernel::calcJacobian<double, 6, false>(dim3, dim3, cudaStream_t,
    const double*, const double*, const unsigned*, const double*,
    const double*, double, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, double*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<float>, 3, true>(dim3, dim3, cudaStream_t,
    const thrust::complex<float>*, const thrust::complex<float>*, const unsigned*, const thrust::complex<float>*,
    const thrust::complex<float>*, thrust::complex<float>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<float>, 3, false>(dim3, dim3, cudaStream_t,
    const thrust::complex<float>*, const thrust::complex<float>*, const unsigned*, const thrust::complex<float>*,
    const thrust::complex<float>*, thrust::complex<float>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<float>, 6, true>(dim3, dim3, cudaStream_t,
    const thrust::complex<float>*, const thrust::complex<float>*, const unsigned*, const thrust::complex<float>*,
    const thrust::complex<float>*, thrust::complex<float>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<float>, 6, false>(dim3, dim3, cudaStream_t,
    const thrust::complex<float>*, const thrust::complex<float>*, const unsigned*, const thrust::complex<float>*,
    const thrust::complex<float>*, thrust::complex<float>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<float>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<double>, 3, true>(dim3, dim3, cudaStream_t,
    const thrust::complex<double>*, const thrust::complex<double>*, const unsigned*, const thrust::complex<double>*,
    const thrust::complex<double>*, thrust::complex<double>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<double>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<double>, 3, false>(dim3, dim3, cudaStream_t,
    const thrust::complex<double>*, const thrust::complex<double>*, const unsigned*, const thrust::complex<double>*,
    const thrust::complex<double>*, thrust::complex<double>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<double>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<double>, 6, true>(dim3, dim3, cudaStream_t,
    const thrust::complex<double>*, const thrust::complex<double>*, const unsigned*, const thrust::complex<double>*,
    const thrust::complex<double>*, thrust::complex<double>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<double>*);
template void mpFlow::FEM::equationKernel::calcJacobian<thrust::complex<double>, 6, false>(dim3, dim3, cudaStream_t,
    const thrust::complex<double>*, const thrust::complex<double>*, const unsigned*, const thrust::complex<double>*,
    const thrust::complex<double>*, thrust::complex<double>, unsigned, unsigned, unsigned, unsigned,
    unsigned, unsigned, bool, thrust::complex<double>*);
