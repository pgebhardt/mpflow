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
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/numeric/matrix_kernel.h"

// fill kernel
template <
    class type
>
__global__ void fillKernel(const type value, mpFlow::dtype::size rows, type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    result[row + column * rows] = value;
}

// fill kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::fill(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type value, mpFlow::dtype::size rows, type* result) {
    // call cuda kernel
    fillKernel<type><<<blocks, threads, 0, stream>>>(value, rows, result);

    CudaCheckError();
}

// fill specialisation
template void mpFlow::numeric::matrixKernel::fill<mpFlow::dtype::real>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::real,
    mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::fill<mpFlow::dtype::index>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::index,
    mpFlow::dtype::size, mpFlow::dtype::index*);

// add kernel
template <
    class type
>
__global__ void addKernel(const type* matrix, mpFlow::dtype::size rows, type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    result[row + column * rows] += matrix[row + column * rows];
}

// add kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::add(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* matrix, mpFlow::dtype::size rows, type* result) {
    // call cuda kernel
    addKernel<type><<<blocks, threads, 0, stream>>>(matrix, rows, result);

    CudaCheckError();
}

// add specialisation
template void mpFlow::numeric::matrixKernel::add<mpFlow::dtype::real>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::real*,
    mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::add<mpFlow::dtype::index>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index*);

// scale kernel
template <
    class type
>
__global__ void scaleKernel(type scalar, mpFlow::dtype::size rows, type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // scale matrix with scalar
    result[row + column * rows] *= scalar;
}

// scale kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::scale(dim3 blocks, dim3 threads, cudaStream_t stream,
    type scalar, dtype::size rows, type* result) {
    // call cuda kernel
    scaleKernel<type><<<blocks, threads, 0, stream>>>(scalar, rows, result);

    CudaCheckError();
}

// scale specialisation
template void mpFlow::numeric::matrixKernel::scale<mpFlow::dtype::real>(
    dim3, dim3, cudaStream_t, mpFlow::dtype::real, mpFlow::dtype::size,
    mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::scale<mpFlow::dtype::index>(
    dim3, dim3, cudaStream_t, mpFlow::dtype::index, mpFlow::dtype::size,
    mpFlow::dtype::index*);

// elementwise multiply kernel
template <
    class type
>
__global__ void elementwiseMultiplyKernel(const type* a, const type* b, mpFlow::dtype::size rows,
    type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

// elementwise multiply kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::elementwiseMultiply(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* a, const type* b, dtype::size rows,
    type* result) {
    // call cuda kernel
    elementwiseMultiplyKernel<type><<<blocks, threads, 0, stream>>>(
        a, b, rows, result);

    CudaCheckError();
}

// elementwise multiply specialisation
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<mpFlow::dtype::real>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::real*,
    const mpFlow::dtype::real*, mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<mpFlow::dtype::index>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::index*,
    const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::index*);

// elementwise division kernel
template <
    class type
>
__global__ void elementwiseDivisionKernel(const type* a, const type* b, mpFlow::dtype::size rows,
    type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise division
    result[row + column * rows] = a[row + column * rows] / b[row + column * rows];
}

// elementwise multiply kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::elementwiseDivision(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* a, const type* b, dtype::size rows,
    type* result) {
    // call cuda kernel
    elementwiseDivisionKernel<type><<<blocks, threads, 0, stream>>>(
        a, b, rows, result);

    CudaCheckError();
}

// elementwise multiply specialisation
template void mpFlow::numeric::matrixKernel::elementwiseDivision<mpFlow::dtype::real>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::real*,
    const mpFlow::dtype::real*, mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::elementwiseDivision<mpFlow::dtype::index>(
    dim3, dim3, cudaStream_t, const mpFlow::dtype::index*,
    const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::index*);

// sum kernel
template <
    class type
>
__global__ void sumKernel(const type* vector, mpFlow::dtype::size rows, mpFlow::dtype::size offset,
    type* result) {
    // get column
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    mpFlow::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[mpFlow::numeric::matrix::block_size * mpFlow::numeric::matrix::block_size];
    res[lid + threadIdx.y * mpFlow::numeric::matrix::block_size] =
        gid * offset < rows ? vector[gid * offset + column * rows] : 0.0f;

    // reduce
    res[lid + threadIdx.y * mpFlow::numeric::matrix::block_size] +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * mpFlow::numeric::matrix::block_size] : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::block_size] +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * mpFlow::numeric::matrix::block_size] : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::block_size] +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * mpFlow::numeric::matrix::block_size] : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::block_size] +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * mpFlow::numeric::matrix::block_size] : 0.0f;
    __syncthreads();

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows] = res[lid + threadIdx.y * mpFlow::numeric::matrix::block_size];
}

// sum kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::sum(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, dtype::size rows, dtype::size offset, type* result) {
    // call cuda kernel
    sumKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// sum specialisation
template void mpFlow::numeric::matrixKernel::sum<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::sum<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::index*);

// min kernel
template <
    class type
>
__global__ void minKernel(const type* vector, mpFlow::dtype::size rows, mpFlow::dtype::size offset, type* result) {
    // get id
    mpFlow::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[mpFlow::numeric::matrix::block_size];
    res[lid] = gid * offset < rows ? vector[gid * offset] : NAN;

    // reduce
    res[lid] = (lid % 2 == 0) ? (res[lid + 1] < res[lid] ? res[lid + 1] : res[lid]) : res[lid];
    res[lid] = (lid % 4 == 0) ? (res[lid + 2] < res[lid] ? res[lid + 2] : res[lid]) : res[lid];
    res[lid] = (lid % 8 == 0) ? (res[lid + 4] < res[lid] ? res[lid + 4] : res[lid]) : res[lid];
    res[lid] = (lid % 16 == 0) ? (res[lid + 8] < res[lid] ? res[lid + 8] : res[lid]) : res[lid];

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[blockIdx.x * blockDim.x * offset] = res[0];
}

// min kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::min(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, dtype::size rows, dtype::size offset, type* result) {
    // call cuda kernel
    minKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// min specialisation
template void mpFlow::numeric::matrixKernel::min<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::min<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::index*);

// max kernel
template <
    class type
>
__global__ void maxKernel(const type* vector, mpFlow::dtype::size rows, mpFlow::dtype::size offset, type* result) {
    // get id
    mpFlow::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[mpFlow::numeric::matrix::block_size];
    res[lid] = gid * offset < rows ? vector[gid * offset] : NAN;

    // reduce
    res[lid] = (lid % 2 == 0) ? (res[lid + 1] > res[lid] ? res[lid + 1] : res[lid]) : res[lid];
    res[lid] = (lid % 4 == 0) ? (res[lid + 2] > res[lid] ? res[lid + 2] : res[lid]) : res[lid];
    res[lid] = (lid % 8 == 0) ? (res[lid + 4] > res[lid] ? res[lid + 4] : res[lid]) : res[lid];
    res[lid] = (lid % 16 == 0) ? (res[lid + 8] > res[lid] ? res[lid + 8] : res[lid]) : res[lid];

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[blockIdx.x * blockDim.x * offset] = res[0];
}

// max kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::max(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, dtype::size rows, dtype::size offset, type* result) {
    // call cuda kernel
    maxKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// max specialisation
template void mpFlow::numeric::matrixKernel::max<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::real*);
template void mpFlow::numeric::matrixKernel::max<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::index*);
