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
#include "fasteit/matrix_kernel.h"


// add kernel
template <
    class type
>
__global__ void addKernel(const type* matrix, fastEIT::dtype::size rows, type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    result[row + column * rows] += matrix[row + column * rows];
}

// add kernel wrapper
template <
    class type
>
void fastEIT::matrixKernel::add(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* matrix, fastEIT::dtype::size rows, type* result) {
    // call cuda kernel
    addKernel<type><<<blocks, threads, 0, stream>>>(matrix, rows, result);

    CudaCheckError();
}

// add specialisation
template void fastEIT::matrixKernel::add<fastEIT::dtype::real>(
    dim3, dim3, cudaStream_t, const fastEIT::dtype::real*,
    fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::matrixKernel::add<fastEIT::dtype::index>(
    dim3, dim3, cudaStream_t, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::index*);

// scale kernel
template <
    class type
>
__global__ void scaleKernel(type scalar, fastEIT::dtype::size rows, type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // scale matrix with scalar
    result[row + column * rows] *= scalar;
}

// scale kernel wrapper
template <
    class type
>
void fastEIT::matrixKernel::scale(dim3 blocks, dim3 threads, cudaStream_t stream,
    type scalar, dtype::size rows, type* result) {
    // call cuda kernel
    scaleKernel<type><<<blocks, threads, 0, stream>>>(scalar, rows, result);

    CudaCheckError();
}

// scale specialisation
template void fastEIT::matrixKernel::scale<fastEIT::dtype::real>(
    dim3, dim3, cudaStream_t, fastEIT::dtype::real, fastEIT::dtype::size,
    fastEIT::dtype::real*);
template void fastEIT::matrixKernel::scale<fastEIT::dtype::index>(
    dim3, dim3, cudaStream_t, fastEIT::dtype::index, fastEIT::dtype::size,
    fastEIT::dtype::index*);

// vector dot product kernel
template <
    class type
>
__global__ void vectorDotProductKernel(const type* a, const type* b, fastEIT::dtype::size rows,
    type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

// vector dot product kernel wrapper
template <
    class type
>
void fastEIT::matrixKernel::vectorDotProduct(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* a, const type* b, dtype::size rows,
    type* result) {
    // call cuda kernel
    vectorDotProductKernel<type><<<blocks, threads, 0, stream>>>(
        a, b, rows, result);

    CudaCheckError();
}

// vector dot product specialisation
template void fastEIT::matrixKernel::vectorDotProduct<fastEIT::dtype::real>(
    dim3, dim3, cudaStream_t, const fastEIT::dtype::real*,
    const fastEIT::dtype::real*, fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::matrixKernel::vectorDotProduct<fastEIT::dtype::index>(
    dim3, dim3, cudaStream_t, const fastEIT::dtype::index*,
    const fastEIT::dtype::index*, fastEIT::dtype::size, fastEIT::dtype::index*);

// sum kernel
template <
    class type
>
__global__ void sumKernel(const type* vector, fastEIT::dtype::size rows, fastEIT::dtype::size offset,
    type* result) {
    // get column
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    fastEIT::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[fastEIT::matrix::block_size * fastEIT::matrix::block_size];
    res[lid + threadIdx.y * fastEIT::matrix::block_size] =
        gid * offset < rows ? vector[gid * offset + column * rows] : 0.0f;

    // reduce
    res[lid + threadIdx.y * fastEIT::matrix::block_size] +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * fastEIT::matrix::block_size] : 0.0f;
    res[lid + threadIdx.y * fastEIT::matrix::block_size] +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * fastEIT::matrix::block_size] : 0.0f;
    res[lid + threadIdx.y * fastEIT::matrix::block_size] +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * fastEIT::matrix::block_size] : 0.0f;
    res[lid + threadIdx.y * fastEIT::matrix::block_size] +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * fastEIT::matrix::block_size] : 0.0f;
    __syncthreads();

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows] = res[lid + threadIdx.y * fastEIT::matrix::block_size];
}

// sum kernel wrapper
template <
    class type
>
void fastEIT::matrixKernel::sum(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, dtype::size rows, dtype::size offset, type* result) {
    // call cuda kernel
    sumKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// sum specialisation
template void fastEIT::matrixKernel::sum<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::matrixKernel::sum<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::index*);

// min kernel
template <
    class type
>
__global__ void minKernel(const type* vector, fastEIT::dtype::size rows, fastEIT::dtype::size offset, type* result) {
    // get id
    fastEIT::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[fastEIT::matrix::block_size];
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
void fastEIT::matrixKernel::min(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, dtype::size rows, dtype::size offset, type* result) {
    // call cuda kernel
    minKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// min specialisation
template void fastEIT::matrixKernel::min<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::matrixKernel::min<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::index*);

// max kernel
template <
    class type
>
__global__ void maxKernel(const type* vector, fastEIT::dtype::size rows, fastEIT::dtype::size offset, type* result) {
    // get id
    fastEIT::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[fastEIT::matrix::block_size];
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
void fastEIT::matrixKernel::max(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, dtype::size rows, dtype::size offset, type* result) {
    // call cuda kernel
    maxKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// max specialisation
template void fastEIT::matrixKernel::max<fastEIT::dtype::real>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::real*, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::matrixKernel::max<fastEIT::dtype::index>(dim3, dim3,
    cudaStream_t, const fastEIT::dtype::index*, fastEIT::dtype::size,
    fastEIT::dtype::size, fastEIT::dtype::index*);
