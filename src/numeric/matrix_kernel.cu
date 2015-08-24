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
#include "mpflow/numeric/matrix_kernel.h"

// fill kernel
template <
    class type
>
__global__ void fillKernel(type const value, unsigned const rows,
    unsigned const cols, unsigned const dataRows, type* const result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    result[row + col * dataRows] = (row < rows && col < cols) ? value : type(0);
}

// fill kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::fill(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    type const value, unsigned const rows, unsigned const cols, unsigned const dataRows, type* const result) {
    // call cuda kernel
    fillKernel<type><<<blocks, threads, 0, stream>>>(value, rows, cols, dataRows, result);

    CudaCheckError();
}

// fill specialisation
template void mpFlow::numeric::matrixKernel::fill<float>(
    dim3 const, dim3 const, cudaStream_t const, float const,
    unsigned const, unsigned const, unsigned const, float* const);
template void mpFlow::numeric::matrixKernel::fill<double>(
    dim3 const, dim3 const, cudaStream_t const, double const,
    unsigned const, unsigned const, unsigned const, double* const);
template void mpFlow::numeric::matrixKernel::fill<thrust::complex<float> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<float> const,
    unsigned const, unsigned const, unsigned const, thrust::complex<float>* const);
template void mpFlow::numeric::matrixKernel::fill<thrust::complex<double> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<double> const,
    unsigned const, unsigned const, unsigned const, thrust::complex<double>* const);
template void mpFlow::numeric::matrixKernel::fill<unsigned>(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, unsigned const, unsigned const, unsigned* const);
template void mpFlow::numeric::matrixKernel::fill<int>(
    dim3 const, dim3 const, cudaStream_t const, int const,
    unsigned const, unsigned const, unsigned const, int* const);

// fill unity matrix kernel
template <
    class type
>
__global__ void setEyeKernel(unsigned const rows, unsigned const dataRows,
    type* const matrix) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;

    // set unity matrix
    matrix[row + col * dataRows] = row < rows ? (row == col ? type(1) : type(0)) : type(0);
}

// fill unity matrix
template <
    class type
>
void mpFlow::numeric::matrixKernel::setEye(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    unsigned const rows, unsigned const dataRows, type* const matrix) {
    // call cuda kernel
    setEyeKernel<type><<<blocks, threads, 0, stream>>>(rows, dataRows, matrix);

    CudaCheckError();        
}            

// setEye specialisation
template void mpFlow::numeric::matrixKernel::setEye<float>(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, float* const);
template void mpFlow::numeric::matrixKernel::setEye<double>(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, double* const);
template void mpFlow::numeric::matrixKernel::setEye<thrust::complex<float> >(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, thrust::complex<float>* const);
template void mpFlow::numeric::matrixKernel::setEye<thrust::complex<double> >(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, thrust::complex<double>* const);
template void mpFlow::numeric::matrixKernel::setEye<unsigned>(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, unsigned* const);
template void mpFlow::numeric::matrixKernel::setEye<int>(
    dim3 const, dim3 const, cudaStream_t const, unsigned const,
    unsigned const, int* const);

// create diagonal matrix kernel
template <
    class type
>
__global__ void diagKernel(type const* const matrix, unsigned const dataRows,
    type* const result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;

    // set unity matrix
    result[row + col * dataRows] = row == col ? matrix[row + col * dataRows] : type(0);
}

// create diagonal matrix
template <
    class type
>
void mpFlow::numeric::matrixKernel::diag(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    type const* const matrix, unsigned const dataRows, type* const result) {
    // call cuda kernel
    diagKernel<type><<<blocks, threads, 0, stream>>>(matrix, dataRows, result);

    CudaCheckError();        
}

// diag specialisation
template void mpFlow::numeric::matrixKernel::diag<float>(
    dim3 const, dim3 const, cudaStream_t const, float const* const,
    unsigned const, float* const);
template void mpFlow::numeric::matrixKernel::diag<double>(
    dim3 const, dim3 const, cudaStream_t const, double const* const,
    unsigned const, double* const);
template void mpFlow::numeric::matrixKernel::diag<thrust::complex<float> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<float> const* const,
    unsigned const, thrust::complex<float>* const);
template void mpFlow::numeric::matrixKernel::diag<thrust::complex<double> >(
    dim3 const, dim3 const, cudaStream_t const, thrust::complex<double> const* const,
    unsigned const, thrust::complex<double>* const);
template void mpFlow::numeric::matrixKernel::diag<unsigned>(
    dim3 const, dim3 const, cudaStream_t const, unsigned const* const,
    unsigned const, unsigned* const);
template void mpFlow::numeric::matrixKernel::diag<int>(
    dim3 const, dim3 const, cudaStream_t const, int const* const,
    unsigned const, int* const);

// add scalar kernel
template <class type>
__global__ void addKernel(type const value, unsigned const rows, type* const result) {
    // get ids
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned const col = blockIdx.y * blockDim.y + threadIdx.y;

    result[row + col * rows] += value;
}

// add scalar kernel wrapper
template <class type>
void mpFlow::numeric::matrixKernel::add(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    type const value, unsigned const rows, type* const result) {
    // call cuda kernel
    addKernel<type><<<blocks, threads, 0, stream>>>(value, rows, result);

    CudaCheckError();
}

template void mpFlow::numeric::matrixKernel::add<float>(dim3 const, dim3 const, cudaStream_t const,
    float const, unsigned const, float* const);
template void mpFlow::numeric::matrixKernel::add<double>(dim3 const, dim3 const, cudaStream_t const,
    double const, unsigned const, double* const);
template void mpFlow::numeric::matrixKernel::add<thrust::complex<float> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<float> const, unsigned const, thrust::complex<float>* const);
template void mpFlow::numeric::matrixKernel::add<thrust::complex<double> >(dim3 const, dim3 const, cudaStream_t const,
    thrust::complex<double> const, unsigned const, thrust::complex<double>* const);
template void mpFlow::numeric::matrixKernel::add<unsigned>(dim3 const, dim3 const, cudaStream_t const,
    unsigned const, unsigned const, unsigned* const);
template void mpFlow::numeric::matrixKernel::add<int>(dim3 const, dim3 const, cudaStream_t const,
    int const, unsigned const, int* const);
    
// add kernel
template <
    class type
>
__global__ void addKernel(const type* matrix, unsigned rows, type* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    result[row + column * rows] += matrix[row + column * rows];
}

// add kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::add(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* matrix, unsigned rows, type* result) {
    // call cuda kernel
    addKernel<type><<<blocks, threads, 0, stream>>>(matrix, rows, result);

    CudaCheckError();
}

// add specialisation
template void mpFlow::numeric::matrixKernel::add<float>(
    dim3, dim3, cudaStream_t, const float*,
    unsigned, float*);
template void mpFlow::numeric::matrixKernel::add<double>(
    dim3, dim3, cudaStream_t, const double*,
    unsigned, double*);
template void mpFlow::numeric::matrixKernel::add<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, const thrust::complex<float>*,
    unsigned, thrust::complex<float>*);
template void mpFlow::numeric::matrixKernel::add<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, const thrust::complex<double>*,
    unsigned, thrust::complex<double>*);
template void mpFlow::numeric::matrixKernel::add<unsigned>(
    dim3, dim3, cudaStream_t, const unsigned*,
    unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::add<int>(
    dim3, dim3, cudaStream_t, const int*,
    unsigned, int*);

// scale kernel
template <
    class type
>
__global__ void scaleKernel(type scalar, unsigned rows, type* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // scale matrix with scalar
    result[row + column * rows] *= scalar;
}

// scale kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::scale(dim3 blocks, dim3 threads, cudaStream_t stream,
    type scalar, unsigned rows, type* result) {
    // call cuda kernel
    scaleKernel<type><<<blocks, threads, 0, stream>>>(scalar, rows, result);

    CudaCheckError();
}

// scale specialisation
template void mpFlow::numeric::matrixKernel::scale<float>(
    dim3, dim3, cudaStream_t, float, unsigned,
    float*);
template void mpFlow::numeric::matrixKernel::scale<double>(
    dim3, dim3, cudaStream_t, double, unsigned,
    double*);
template void mpFlow::numeric::matrixKernel::scale<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, thrust::complex<float>, unsigned,
    thrust::complex<float>*);
template void mpFlow::numeric::matrixKernel::scale<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, thrust::complex<double>, unsigned,
    thrust::complex<double>*);
template void mpFlow::numeric::matrixKernel::scale<unsigned>(
    dim3, dim3, cudaStream_t, unsigned, unsigned,
    unsigned*);
template void mpFlow::numeric::matrixKernel::scale<int>(
    dim3, dim3, cudaStream_t, int, unsigned,
    int*);

// elementwise multiply kernel
template <
    class type
>
__global__ void elementwiseMultiplyKernel(const type* a, const type* b, unsigned rows,
    type* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

// elementwise multiply kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::elementwiseMultiply(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* a, const type* b, unsigned rows,
    type* result) {
    // call cuda kernel
    elementwiseMultiplyKernel<<<blocks, threads, 0, stream>>>(
        a, b, rows, result);

    CudaCheckError();
}

// elementwise multiply specialisation
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<float>(
    dim3, dim3, cudaStream_t, const float*,
    const float*, unsigned, float*);
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<double>(
    dim3, dim3, cudaStream_t, const double*,
    const double*, unsigned, double*);
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, const thrust::complex<float>*,
    const thrust::complex<float>*, unsigned, thrust::complex<float>*);
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, const thrust::complex<double>*,
    const thrust::complex<double>*, unsigned, thrust::complex<double>*);
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<unsigned>(
    dim3, dim3, cudaStream_t, const unsigned*,
    const unsigned*, unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::elementwiseMultiply<int>(
    dim3, dim3, cudaStream_t, const int*,
    const int*, unsigned, int*);

// elementwise division kernel
template <
    class type
>
__global__ void elementwiseDivisionKernel(const type* a, const type* b, unsigned rows,
    type* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned col = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise division
    result[row + col * rows] = b[row + col * rows] != type(0) ? a[row + col * rows] / b[row + col * rows] : type(0);
}

// elementwise division kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::elementwiseDivision(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* a, const type* b, unsigned rows,
    type* result) {
    // call cuda kernel
    elementwiseDivisionKernel<type><<<blocks, threads, 0, stream>>>(
        a, b, rows, result);

    CudaCheckError();
}

// elementwise division specialisation
template void mpFlow::numeric::matrixKernel::elementwiseDivision<float>(
    dim3, dim3, cudaStream_t, const float*,
    const float*, unsigned, float*);
template void mpFlow::numeric::matrixKernel::elementwiseDivision<double>(
    dim3, dim3, cudaStream_t, const double*,
    const double*, unsigned, double*);
template void mpFlow::numeric::matrixKernel::elementwiseDivision<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, const thrust::complex<float>*,
    const thrust::complex<float>*, unsigned, thrust::complex<float>*);
template void mpFlow::numeric::matrixKernel::elementwiseDivision<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, const thrust::complex<double>*,
    const thrust::complex<double>*, unsigned, thrust::complex<double>*);
template void mpFlow::numeric::matrixKernel::elementwiseDivision<unsigned>(
    dim3, dim3, cudaStream_t, const unsigned*,
    const unsigned*, unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::elementwiseDivision<int>(
    dim3, dim3, cudaStream_t, const int*,
    const int*, unsigned, int*);

// vectorDotProduct kernel
template <
    class type
>
__global__ void vectorDotProductKernel(const type* a, const type* b, unsigned rows,
    type* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

template <
    class type
>
__global__ void vectorDotProductKernel(const thrust::complex<type>* a, const thrust::complex<type>* b,
    unsigned rows, thrust::complex<type>* result) {
    // get ids
    unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * conj(b[row + column * rows]);
}

// vectorDotProduct kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::vectorDotProduct(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* a, const type* b, unsigned rows,
    type* result) {
    // call cuda kernel
    vectorDotProductKernel<<<blocks, threads, 0, stream>>>(
        a, b, rows, result);

    CudaCheckError();
}

// elementwise multiply specialisation
template void mpFlow::numeric::matrixKernel::vectorDotProduct<float>(
    dim3, dim3, cudaStream_t, const float*,
    const float*, unsigned, float*);
template void mpFlow::numeric::matrixKernel::vectorDotProduct<double>(
    dim3, dim3, cudaStream_t, const double*,
    const double*, unsigned, double*);
template void mpFlow::numeric::matrixKernel::vectorDotProduct<thrust::complex<float> >(
    dim3, dim3, cudaStream_t, const thrust::complex<float>*,
    const thrust::complex<float>*, unsigned, thrust::complex<float>*);
template void mpFlow::numeric::matrixKernel::vectorDotProduct<thrust::complex<double> >(
    dim3, dim3, cudaStream_t, const thrust::complex<double>*,
    const thrust::complex<double>*, unsigned, thrust::complex<double>*);
template void mpFlow::numeric::matrixKernel::vectorDotProduct<unsigned>(
    dim3, dim3, cudaStream_t, const unsigned*,
    const unsigned*, unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::vectorDotProduct<int>(
    dim3, dim3, cudaStream_t, const int*,
    const int*, unsigned, int*);

template <class type>
__global__ void setIndexedElementsKernel(unsigned const* const indices, unsigned const indicesRows,
    type const value, unsigned const rows, type* const result) {
    // get ids
    unsigned const row = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned const col = blockIdx.y * blockDim.y + threadIdx.y;
    
    unsigned const index = indices[row + col * indicesRows];
    if (index != mpFlow::constants::invalidIndex) {
        result[index + col * rows] = value;
    }
}

// set indexed elements kernel wrapper
template <class type>
void mpFlow::numeric::matrixKernel::setIndexedElements(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
    unsigned const* const indices, unsigned const indicesRows, type const value,
    unsigned const rows, type* const result) {
    // call cuda kernel
    setIndexedElementsKernel<type><<<blocks, threads, 0, stream>>>(indices, indicesRows, value,
        rows, result);
    
    CudaCheckError();   
}

template void mpFlow::numeric::matrixKernel::setIndexedElements<float>(dim3 const, dim3 const, cudaStream_t const,
    unsigned const* const, unsigned const, float const, unsigned const, float* const);
template void mpFlow::numeric::matrixKernel::setIndexedElements<double>(dim3 const, dim3 const, cudaStream_t const,
    unsigned const* const, unsigned const, double const, unsigned const, double* const);
template void mpFlow::numeric::matrixKernel::setIndexedElements<thrust::complex<float> >(dim3 const, dim3 const, cudaStream_t const,
    unsigned const* const, unsigned const, thrust::complex<float> const, unsigned const, thrust::complex<float>* const);
template void mpFlow::numeric::matrixKernel::setIndexedElements<thrust::complex<double> >(dim3 const, dim3 const, cudaStream_t const,
    unsigned const* const, unsigned const, thrust::complex<double> const, unsigned const, thrust::complex<double>* const);
template void mpFlow::numeric::matrixKernel::setIndexedElements<unsigned>(dim3 const, dim3 const, cudaStream_t const,
    unsigned const* const, unsigned const, unsigned const, unsigned const, unsigned* const);
template void mpFlow::numeric::matrixKernel::setIndexedElements<int>(dim3 const, dim3 const, cudaStream_t const,
    unsigned const* const, unsigned const, int const, unsigned const, int* const);
    
// sum kernel
template <
    class type
>
__global__ void sumKernel(const type* vector, unsigned rows, unsigned offset,
    type* result) {
    // get column
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lid = threadIdx.x;

    // copy data to shared memory
    volatile __shared__ type res[mpFlow::numeric::matrix::blockSize * mpFlow::numeric::matrix::blockSize];
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize] =
        gid * offset < rows ? vector[gid * offset + column * rows] : 0.0f;

    // reduce
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize] +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * mpFlow::numeric::matrix::blockSize] : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize] +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * mpFlow::numeric::matrix::blockSize] : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize] +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * mpFlow::numeric::matrix::blockSize] : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize] +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * mpFlow::numeric::matrix::blockSize] : 0.0f;
    __syncthreads();

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows] = res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize];
}

// complex specialisation
template <>
__global__ void sumKernel(const thrust::complex<float>* vector, unsigned rows,
    unsigned offset, thrust::complex<float>* result) {
    // get column
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lid = threadIdx.x;

    // copy data to shared memory
    volatile __shared__ cuFloatComplex res[mpFlow::numeric::matrix::blockSize * mpFlow::numeric::matrix::blockSize];
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x =
        gid * offset < rows ? vector[gid * offset + column * rows].real() : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y =
        gid * offset < rows ? vector[gid * offset + column * rows].imag() : 0.0f;

    // reduce
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows].real(res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x);
    result[gid * offset + column * rows].imag(res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y);
}

template <>
__global__ void sumKernel(const thrust::complex<double>* vector, unsigned rows,
    unsigned offset, thrust::complex<double>* result) {
    // get column
    unsigned column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lid = threadIdx.x;

    // copy data to shared memory
    volatile __shared__ cuDoubleComplex res[mpFlow::numeric::matrix::blockSize * mpFlow::numeric::matrix::blockSize];
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x =
        gid * offset < rows ? vector[gid * offset + column * rows].real() : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y =
        gid * offset < rows ? vector[gid * offset + column * rows].imag() : 0.0f;

    // reduce
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * mpFlow::numeric::matrix::blockSize].x : 0.0f;
    res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y +=
        (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * mpFlow::numeric::matrix::blockSize].y : 0.0f;

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows].real(res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].x);
    result[gid * offset + column * rows].imag(res[lid + threadIdx.y * mpFlow::numeric::matrix::blockSize].y);
}

// sum kernel wrapper
template <
    class type
>
void mpFlow::numeric::matrixKernel::sum(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* vector, unsigned rows, unsigned offset, type* result) {
    // call cuda kernel
    sumKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// sum specialisation
template void mpFlow::numeric::matrixKernel::sum<float>(dim3, dim3,
    cudaStream_t, const float*, unsigned,
    unsigned, float*);
template void mpFlow::numeric::matrixKernel::sum<double>(dim3, dim3,
    cudaStream_t, const double*, unsigned,
    unsigned, double*);
template void mpFlow::numeric::matrixKernel::sum<thrust::complex<float> >(dim3, dim3,
    cudaStream_t, const thrust::complex<float>*, unsigned,
    unsigned, thrust::complex<float>*);
template void mpFlow::numeric::matrixKernel::sum<thrust::complex<double> >(dim3, dim3,
    cudaStream_t, const thrust::complex<double>*, unsigned,
    unsigned, thrust::complex<double>*);
template void mpFlow::numeric::matrixKernel::sum<unsigned>(dim3, dim3,
    cudaStream_t, const unsigned*, unsigned,
    unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::sum<int>(dim3, dim3,
    cudaStream_t, const int*, unsigned,
    unsigned, int*);

// min kernel
template <
    class type
>
__global__ void minKernel(const type* vector, unsigned rows, unsigned offset, type* result) {
    // get id
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lid = threadIdx.x;

    // copy data to shared memory
    volatile __shared__ type res[mpFlow::numeric::matrix::blockSize];
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
    const type* vector, unsigned rows, unsigned offset, type* result) {
    // call cuda kernel
    minKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// min specialisation
template void mpFlow::numeric::matrixKernel::min<float>(dim3, dim3,
    cudaStream_t, const float*, unsigned,
    unsigned, float*);
template void mpFlow::numeric::matrixKernel::min<double>(dim3, dim3,
    cudaStream_t, const double*, unsigned,
    unsigned, double*);
template void mpFlow::numeric::matrixKernel::min<unsigned>(dim3, dim3,
    cudaStream_t, const unsigned*, unsigned,
    unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::min<int>(dim3, dim3,
    cudaStream_t, const int*, unsigned,
    unsigned, int*);

// max kernel
template <
    class type
>
__global__ void maxKernel(const type* vector, unsigned rows, unsigned offset, type* result) {
    // get id
    unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned lid = threadIdx.x;

    // copy data to shared memory
    volatile __shared__ type res[mpFlow::numeric::matrix::blockSize];
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
    const type* vector, unsigned rows, unsigned offset, type* result) {
    // call cuda kernel
    maxKernel<type><<<blocks, threads, 0, stream>>>(vector, rows, offset, result);

    CudaCheckError();
}

// max specialisation
template void mpFlow::numeric::matrixKernel::max<float>(dim3, dim3,
    cudaStream_t, const float*, unsigned,
    unsigned, float*);
template void mpFlow::numeric::matrixKernel::max<double>(dim3, dim3,
    cudaStream_t, const double*, unsigned,
    unsigned, double*);
template void mpFlow::numeric::matrixKernel::max<unsigned>(dim3, dim3,
    cudaStream_t, const unsigned*, unsigned,
    unsigned, unsigned*);
template void mpFlow::numeric::matrixKernel::max<int>(dim3, dim3,
    cudaStream_t, const int*, unsigned,
    unsigned, int*);
