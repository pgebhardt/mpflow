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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include "mpflow/cuda_error.h"

#include "mpflow/constants.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/numeric/conjugate_gradient_kernel.h"

// add scalar kernel
template <
    class dataType
>
static __global__ void addScalarKernel(const dataType* scalar,
    unsigned vectorRows, unsigned rows,
    unsigned columns, dataType* vector) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // add data
    vector[row + column * vectorRows] += row < rows && column < columns ?
        scalar[column * vectorRows] : dataType(0);
}

// add scalar kernel wrapper
template <
    class dataType
>
void mpFlow::numeric::conjugateGradientKernel::addScalar(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dataType* scalar, unsigned vector_rows,
    unsigned rows, unsigned columns, dataType* vector) {
    // call cuda kernel
    addScalarKernel<<<blocks, threads, 0, stream>>>(scalar, vector_rows,
        rows, columns, vector);

    CudaCheckError();
}

template void mpFlow::numeric::conjugateGradientKernel::addScalar<float>(
    dim3, dim3, cudaStream_t, const float*, unsigned,
    unsigned, unsigned, float*);
template void mpFlow::numeric::conjugateGradientKernel::addScalar<double>(
    dim3, dim3, cudaStream_t, const double*, unsigned,
    unsigned, unsigned, double*);
template void mpFlow::numeric::conjugateGradientKernel::addScalar<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, const thrust::complex<float>*, unsigned,
    unsigned, unsigned, thrust::complex<float>*);
template void mpFlow::numeric::conjugateGradientKernel::addScalar<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, const thrust::complex<double>*, unsigned,
    unsigned, unsigned, thrust::complex<double>*);

// update vector kernel
template <
    class dataType
>
static __global__ void updateVectorKernel(const dataType* x1,
    const double sign, const dataType* x2,
    const dataType* r1, const dataType* r2,
    unsigned rows, dataType* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = r2[column * rows] != dataType(0) ?
        x1[row + column * rows] + dataType(sign) * x2[row + column * rows] * r1[column * rows] / r2[column * rows] :
        dataType(0);
}

// update vector kernel wrapper
template <
    class dataType
>
void mpFlow::numeric::conjugateGradientKernel::updateVector(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dataType* x1, const double sign,
    const dataType* x2, const dataType* r1, const dataType* r2,
    unsigned rows, dataType* result) {
    // call cuda kernel
    updateVectorKernel<<<blocks, threads, 0, stream>>>(x1, sign, x2, r1, r2, rows,
        result);

    CudaCheckError();
}

template void mpFlow::numeric::conjugateGradientKernel::updateVector<float>(
    dim3, dim3, cudaStream_t, const float*, const double,
    const float*, const float*, const float*,
    unsigned, float*);
template void mpFlow::numeric::conjugateGradientKernel::updateVector<double>(
    dim3, dim3, cudaStream_t, const double*, const double,
    const double*, const double*, const double*,
    unsigned, double*);
template void mpFlow::numeric::conjugateGradientKernel::updateVector<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, const thrust::complex<float>*, const double,
    const thrust::complex<float>*, const thrust::complex<float>*, const thrust::complex<float>*,
    unsigned, thrust::complex<float>*);
template void mpFlow::numeric::conjugateGradientKernel::updateVector<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, const thrust::complex<double>*, const double,
    const thrust::complex<double>*, const thrust::complex<double>*, const thrust::complex<double>*,
    unsigned, thrust::complex<double>*);
