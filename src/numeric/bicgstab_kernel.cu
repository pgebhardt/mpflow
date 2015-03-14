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

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/numeric/bicgstab_kernel.h"

// update vector kernel
template <
    class dataType
>
static __global__ void updateVectorKernel(const dataType* x1,
    const double sign, const dataType* x2, const dataType* scalar,
    mpFlow::dtype::size rows, dataType* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc value
    result[row + column * rows] = x1[row + column * rows] +
        dataType(sign) * scalar[column * rows] * x2[row + column * rows];
}

// update vector kernel wrapper
template <
    class dataType
>
void mpFlow::numeric::bicgstabKernel::updateVector(dim3 blocks, dim3 threads,
    cudaStream_t stream, const dataType* x1, const double sign,
    const dataType* x2, const dataType* scalar, dtype::size rows,
    dataType* result) {
    // call cuda kernel
    updateVectorKernel<<<blocks, threads, 0, stream>>>(x1, sign, x2, scalar, rows,
        result);

    CudaCheckError();
}

template void mpFlow::numeric::bicgstabKernel::updateVector<float>(dim3, dim3,
    cudaStream_t, const float*, const double,
    const float*, const float*, mpFlow::dtype::size, float*);
template void mpFlow::numeric::bicgstabKernel::updateVector<double>(dim3, dim3,
    cudaStream_t, const double*, const double,
    const double*, const double*, mpFlow::dtype::size, double*);
template void mpFlow::numeric::bicgstabKernel::updateVector<thrust::complex<float> >(dim3, dim3,
    cudaStream_t, const thrust::complex<float>*, const double,
    const thrust::complex<float>*, const thrust::complex<float>*, mpFlow::dtype::size,
    thrust::complex<float>*);
template void mpFlow::numeric::bicgstabKernel::updateVector<thrust::complex<double> >(dim3, dim3,
    cudaStream_t, const thrust::complex<double>*, const double,
    const thrust::complex<double>*, const thrust::complex<double>*, mpFlow::dtype::size,
    thrust::complex<double>*);
