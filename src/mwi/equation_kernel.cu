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

#include "mpflow/constants.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/mwi/equation_kernel.h"

// calc jacobian kernel
template <
    class dataType
>
static __global__ void calcJacobianKernel(dataType const* const field,
    int const* const connectivityMatrix, dataType const* const elementalJacobianMatrix,
    unsigned rows, unsigned columns, unsigned fieldRows, unsigned elementCount,
    unsigned driveCount, dataType* jacobian) {
    // get id
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned const column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    unsigned const roundMeasurmentCount = (
        (driveCount + mpFlow::numeric::matrix::blockSize - 1) /
        mpFlow::numeric::matrix::blockSize) *
        mpFlow::numeric::matrix::blockSize;
    unsigned const measurmentId = row % roundMeasurmentCount;
    unsigned const driveId = row / roundMeasurmentCount;

    // get data
    dataType dField[3], mField[3];
    for (unsigned i = 0; i < 3; i++) {
        unsigned const index = connectivityMatrix[column + i * columns];
        dField[i] = driveId < driveCount ? field[index + driveId * fieldRows] : dataType(0);
        mField[i] = measurmentId < driveCount ? field[index +
            measurmentId * fieldRows] : dataType(0);
    }

    // calc matrix element
    dataType element = dataType(0);
    for (unsigned i = 0; i < 3; i++)
    for (unsigned j = 0; j < 3; j++) {
        element += dField[i] * mField[j] * elementalJacobianMatrix[column +
            (i + j * 3) * columns];
    }

    jacobian[row + column * rows] = element;
}

// calc jacobian kernel wrapper
template <
    class dataType
>
void mpFlow::MWI::equationKernel::calcJacobian(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    dataType const* const field, int const* const connectivityMatrix,
    dataType const* const elementalJacobianMatrix, unsigned const rows, unsigned const columns,
    unsigned const fieldRows, unsigned const elementCount, unsigned const driveCount, dataType* const jacobian) {
    // call cuda kernel
    calcJacobianKernel<dataType><<<blocks, threads, 0, stream>>>(
        field, connectivityMatrix, elementalJacobianMatrix,
        rows, columns, fieldRows, elementCount, driveCount,
        jacobian);

    CudaCheckError();
}

// template specialisation
template void mpFlow::MWI::equationKernel::calcJacobian<float>(dim3 const, dim3 const, cudaStream_t const,
    float const* const, int const* const, float const* const, unsigned const, unsigned const,
    unsigned const, unsigned const, unsigned const, float* const);
template void mpFlow::MWI::equationKernel::calcJacobian<double>(dim3 const, dim3 const, cudaStream_t const,
    double const* const, int const* const, double const* const, unsigned const, unsigned const,
    unsigned const, unsigned const, unsigned const, double* const);
template void mpFlow::MWI::equationKernel::calcJacobian<thrust::complex<float> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<float> const* const, int const* const, thrust::complex<float> const* const, unsigned const, unsigned const,
    unsigned const, unsigned const, unsigned const, thrust::complex<float>* const);
template void mpFlow::MWI::equationKernel::calcJacobian<thrust::complex<double> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<double> const* const, int const* const, thrust::complex<double> const* const, unsigned const, unsigned const,
    unsigned const, unsigned const, unsigned const, thrust::complex<double>* const);