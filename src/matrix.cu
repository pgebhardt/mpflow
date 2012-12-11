// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix_cuda.h"


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
}

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

/*// sum kernel
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
    __volatile __shared__ type res[fastEIT::Matrix<type>::block_size * fastEIT::Matrix<type>::block_size];
    res[lid + threadIdx.y * fastEIT::Matrix<type>::block_size] =
        gid * offset < rows ? vector[gid * offset + column * rows] : 0.0f;

    // reduce
    res[lid + threadIdx.y * fastEIT::Matrix<type>::block_size] +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * fastEIT::Matrix<type>::block_size] : 0.0f;
    res[lid + threadIdx.y * fastEIT::Matrix<type>::block_size] +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * fastEIT::Matrix<type>::block_size] : 0.0f;
    res[lid + threadIdx.y * fastEIT::Matrix<type>::block_size] +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * fastEIT::Matrix<type>::block_size] : 0.0f;
    res[lid + threadIdx.y * fastEIT::Matrix<type>::block_size] +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * fastEIT::Matrix<type>::block_size] : 0.0f;
    __syncthreads();

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows] = res[lid + threadIdx.y * fastEIT::Matrix<type>::block_size];
}

// min kernel
template <
    class type
>
__global__ void minKernel(const type* vector, fastEIT::dtype::size rows, fastEIT::dtype::size offset, type* result) {
    // get id
    fastEIT::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[fastEIT::Matrix<type>::block_size];
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

// max kernel
template <
    class type
>
__global__ void maxKernel(const type* vector, fastEIT::dtype::size rows, fastEIT::dtype::size offset, type* result) {
    // get id
    fastEIT::dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[fastEIT::Matrix<type>::block_size];
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
}*/

// specialisation
template void fastEIT::matrixKernel::add<fastEIT::dtype::real>(
    dim3, dim3, cudaStream_t, const fastEIT::dtype::real*,
    fastEIT::dtype::size, fastEIT::dtype::real*);
template void fastEIT::matrixKernel::add<fastEIT::dtype::index>(
    dim3, dim3, cudaStream_t, const fastEIT::dtype::index*,
    fastEIT::dtype::size, fastEIT::dtype::index*);
