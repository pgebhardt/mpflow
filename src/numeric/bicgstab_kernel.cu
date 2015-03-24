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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include "mpflow/cuda_error.h"

#include "mpflow/constants.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/numeric/bicgstab_kernel.h"

// update vector kernel
template <class dataType>
__global__ void updateVectorKernel(dataType const* const x1,
    double const sign, dataType const* const scalar, dataType const* const x2,
    unsigned const rows, dataType* const result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + col * rows] = x1[row + col * rows] +
        dataType(sign) * scalar[col * rows] * x2[row + col * rows];
}

// update vector kernel wrapper
template <class dataType>
void mpFlow::numeric::bicgstabKernel::updateVector(
    dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    dataType const* const x1, double const sign, dataType const* const scalar,
    dataType const* const x2, unsigned const rows, dataType* const result) {
    // call cuda kernel
    updateVectorKernel<<<blocks, threads, 0, stream>>>(x1, sign, scalar, x2, rows, result);

    CudaCheckError();
}

template void mpFlow::numeric::bicgstabKernel::updateVector<float>(dim3 const, dim3 const,
    cudaStream_t const, float const* const, double const,
    float const* const, float const* const, unsigned const, float* const);
template void mpFlow::numeric::bicgstabKernel::updateVector<double>(dim3 const, dim3 const,
    cudaStream_t const, double const* const, double const,
    double const* const, double const* const, unsigned const, double* const);
template void mpFlow::numeric::bicgstabKernel::updateVector<thrust::complex<float> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<float> const* const,
    double const, thrust::complex<float> const* const, thrust::complex<float> const* const,
    unsigned const, thrust::complex<float>* const);
template void mpFlow::numeric::bicgstabKernel::updateVector<thrust::complex<double> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<double> const* const,
    double const, thrust::complex<double> const* const, thrust::complex<double> const* const,
    unsigned const, thrust::complex<double>* const);
