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
#include "mpflow/numeric/sparse_matrix_kernel.h"

// convert to sparse matrix kernel
template <
    class type
>
static __global__ void convertKernel(const type* matrix,
    mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    type* values, mpFlow::dtype::index* columnIds,
    mpFlow::dtype::index* elementCount) {
    // get id
    mpFlow::dtype::index i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    mpFlow::dtype::size count = 0;

    // init values and columnIds
    for (mpFlow::dtype::index j = 0; j < mpFlow::numeric::sparseMatrix::block_size; j++) {
        values[i * mpFlow::numeric::sparseMatrix::block_size + j] = 0.0f;
        columnIds[i * mpFlow::numeric::sparseMatrix::block_size + j] = mpFlow::dtype::invalid_index;
    }

    // search non-zero elements
    type element = 0.0f;
    for (mpFlow::dtype::index j = 0; j < columns; j++) {
        // get element
        element = matrix[i + j * rows];

        // check for non-zero
        if (element != type(0)) {
            values[i * mpFlow::numeric::sparseMatrix::block_size + count] = element;
            columnIds[i * mpFlow::numeric::sparseMatrix::block_size + count] = j;

            // increment count
            count++;

            // check count
            if (count >= mpFlow::numeric::sparseMatrix::block_size) {
                break;
            }
        }
    }

    // save element count
    elementCount[i] = count;
}

// convert to sparse matrix kernel wrapper
template <
    class type
>
void mpFlow::numeric::sparseMatrixKernel::convert(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* matrix, dtype::size rows, dtype::size columns,
    type* values, dtype::index* columnIds, dtype::index* elementCount) {
    // call cuda kernel
    convertKernel<type><<<blocks, threads, 0, stream>>>(matrix, rows, columns,
        values, columnIds, elementCount);

    CudaCheckError();
}

// convert to matrix kernel
template <
    class type
>
static __global__ void convertToMatrixKernel(const type* values,
    const mpFlow::dtype::index* column_ids, mpFlow::dtype::size density,
    mpFlow::dtype::size rows, type* matrix) {
    // get row id
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // expand sparse matrix
    mpFlow::dtype::index column_id = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0; column < density; ++column) {
        // get column id
        column_id = column_ids[row * mpFlow::numeric::sparseMatrix::block_size + column];

        // set matrix value
        if (column_id != mpFlow::dtype::invalid_index) {
            matrix[row + column_id * rows] = values[
                row * mpFlow::numeric::sparseMatrix::block_size + column];
        }
    }
}

// convert to matrix kernel wrapper
template <
    class type
>
void mpFlow::numeric::sparseMatrixKernel::convertToMatrix(dim3 blocks, dim3 threads,
    cudaStream_t stream, const type* values, const dtype::index* column_ids,
    dtype::size density, dtype::size rows, type* matrix) {
    // call cuda kernel
    convertToMatrixKernel<type><<<blocks, threads, 0, stream>>>(values, column_ids,
        density, rows, matrix);

    CudaCheckError();
}

// sparse matrix multiply kernel
template <
    class type
>
static __global__ void multiplyKernel(const type* values,
    const mpFlow::dtype::index* columnIds, const type* matrix,
    mpFlow::dtype::size result_rows, mpFlow::dtype::size matrix_rows,
    mpFlow::dtype::size columns, mpFlow::dtype::size density, type* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    type res = 0.0f;
    mpFlow::dtype::index id = mpFlow::dtype::invalid_index;

    // read column ids to local memory
    __shared__ mpFlow::dtype::index columnId[
        mpFlow::numeric::sparseMatrix::block_size * mpFlow::numeric::sparseMatrix::block_size];
    __shared__ type value[
        mpFlow::numeric::sparseMatrix::block_size * mpFlow::numeric::sparseMatrix::block_size];

    columnId[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        columnIds[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] : mpFlow::dtype::invalid_index;
    value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        values[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= result_rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (mpFlow::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + j];

         res += id != mpFlow::dtype::invalid_index ? matrix[id + column * matrix_rows] *
            value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + j] : 0.0f;
    }

    // set result
    result[row + column * result_rows] = res;
}

template <>
__global__ void multiplyKernel(const thrust::complex<float>* values,
    const mpFlow::dtype::index* columnIds, const thrust::complex<float>* matrix,
    mpFlow::dtype::size result_rows, mpFlow::dtype::size matrix_rows,
    mpFlow::dtype::size columns, mpFlow::dtype::size density, thrust::complex<float>* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    cuFloatComplex res = make_cuFloatComplex(0.0f, 0.0f);
    mpFlow::dtype::index id = mpFlow::dtype::invalid_index;

    // read column ids to local memory
    __shared__ mpFlow::dtype::index columnId[
        mpFlow::numeric::sparseMatrix::block_size * mpFlow::numeric::sparseMatrix::block_size];
    __shared__ cuFloatComplex value[
        mpFlow::numeric::sparseMatrix::block_size * mpFlow::numeric::sparseMatrix::block_size];

    columnId[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        columnIds[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] : mpFlow::dtype::invalid_index;
    value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].x = row < result_rows ?
        values[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].real() : 0.0f;
    value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].y = row < result_rows ?
        values[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].imag() : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= result_rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (mpFlow::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + j];
        cuFloatComplex element = *(cuFloatComplex*)&matrix[id + column * matrix_rows];
        cuFloatComplex temp = id != mpFlow::dtype::invalid_index ? cuCmulf(element,
            value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + j]) : make_cuFloatComplex(0.0f, 0.0f);

        res.x += temp.x;
        res.y += temp.y;
    }

    // set result
    result[row + column * result_rows].real(res.x);
    result[row + column * result_rows].imag(res.y);
}

template <>
__global__ void multiplyKernel(const thrust::complex<double>* values,
    const mpFlow::dtype::index* columnIds, const thrust::complex<double>* matrix,
    mpFlow::dtype::size result_rows, mpFlow::dtype::size matrix_rows,
    mpFlow::dtype::size columns, mpFlow::dtype::size density, thrust::complex<double>* result) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    cuDoubleComplex res = make_cuDoubleComplex(0.0f, 0.0f);
    mpFlow::dtype::index id = mpFlow::dtype::invalid_index;

    // read column ids to local memory
    __shared__ mpFlow::dtype::index columnId[
        mpFlow::numeric::sparseMatrix::block_size * mpFlow::numeric::sparseMatrix::block_size];
    __shared__ cuDoubleComplex value[
        mpFlow::numeric::sparseMatrix::block_size * mpFlow::numeric::sparseMatrix::block_size];

    columnId[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] = row < result_rows ?
        columnIds[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y] : mpFlow::dtype::invalid_index;
    value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].x = row < result_rows ?
        values[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].real() : 0.0f;
    value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].y = row < result_rows ?
        values[row * mpFlow::numeric::sparseMatrix::block_size + threadIdx.y].imag() : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= result_rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (mpFlow::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + j];
        cuDoubleComplex element = *(cuDoubleComplex*)&matrix[id + column * matrix_rows];
        cuDoubleComplex temp = id != mpFlow::dtype::invalid_index ? cuCmul(element,
            value[threadIdx.x * mpFlow::numeric::sparseMatrix::block_size + j]) : make_cuDoubleComplex(0.0f, 0.0f);

        res.x += temp.x;
        res.y += temp.y;
    }

    // set result
    result[row + column * result_rows].real(res.x);
    result[row + column * result_rows].imag(res.y);
}

// sparse matrix multiply kernel wrapper
template <
    class type
>
void mpFlow::numeric::sparseMatrixKernel::multiply(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* values, const dtype::index* columnIds,
    const type* matrix, dtype::size result_rows, dtype::size matrix_rows,
    dtype::size columns, dtype::size density, type* result) {
    // call cuda kernel
    multiplyKernel<type><<<blocks, threads, 0, stream>>>(values, columnIds, matrix,
        result_rows, matrix_rows, columns, density, result);

    CudaCheckError();
}

// specialisations
// convert to sparse matrix kernel
template void mpFlow::numeric::sparseMatrixKernel::convert<float>(dim3, dim3,
    cudaStream_t, const float*, mpFlow::dtype::size, mpFlow::dtype::size,
    float*, mpFlow::dtype::index*, mpFlow::dtype::index*);
template void mpFlow::numeric::sparseMatrixKernel::convert<double>(dim3, dim3,
    cudaStream_t, const double*, mpFlow::dtype::size, mpFlow::dtype::size,
    double*, mpFlow::dtype::index*, mpFlow::dtype::index*);
template void mpFlow::numeric::sparseMatrixKernel::convert<thrust::complex<float> >(dim3, dim3,
    cudaStream_t, const thrust::complex<float>*, mpFlow::dtype::size, mpFlow::dtype::size,
    thrust::complex<float>*, mpFlow::dtype::index*, mpFlow::dtype::index*);
template void mpFlow::numeric::sparseMatrixKernel::convert<thrust::complex<double> >(dim3, dim3,
    cudaStream_t, const thrust::complex<double>*, mpFlow::dtype::size, mpFlow::dtype::size,
    thrust::complex<double>*, mpFlow::dtype::index*, mpFlow::dtype::index*);
template void mpFlow::numeric::sparseMatrixKernel::convert<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::index*, mpFlow::dtype::index*, mpFlow::dtype::index*);
template void mpFlow::numeric::sparseMatrixKernel::convert<int>(dim3, dim3,
    cudaStream_t, const int*, mpFlow::dtype::size, mpFlow::dtype::size,
    int*, mpFlow::dtype::index*, mpFlow::dtype::index*);

// convertToMatrix kernel
template void mpFlow::numeric::sparseMatrixKernel::convertToMatrix<float>(dim3, dim3,
    cudaStream_t, const float*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, float*);
template void mpFlow::numeric::sparseMatrixKernel::convertToMatrix<double>(dim3, dim3,
    cudaStream_t, const double*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, double*);
template void mpFlow::numeric::sparseMatrixKernel::convertToMatrix<thrust::complex<float> >(dim3, dim3,
    cudaStream_t, const thrust::complex<float>*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, thrust::complex<float>*);
template void mpFlow::numeric::sparseMatrixKernel::convertToMatrix<thrust::complex<double> >(dim3, dim3,
    cudaStream_t, const thrust::complex<double>*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, thrust::complex<double>*);
template void mpFlow::numeric::sparseMatrixKernel::convertToMatrix<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, mpFlow::dtype::index* matrix);
template void mpFlow::numeric::sparseMatrixKernel::convertToMatrix<int>(dim3, dim3,
    cudaStream_t, const int*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::size, int* matrix);

// multiply kernel
template void mpFlow::numeric::sparseMatrixKernel::multiply<float>(dim3, dim3,
    cudaStream_t, const float*, const mpFlow::dtype::index*,
    const float*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, float*);
template void mpFlow::numeric::sparseMatrixKernel::multiply<double>(dim3, dim3,
    cudaStream_t, const double*, const mpFlow::dtype::index*,
    const double*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, double*);
template void mpFlow::numeric::sparseMatrixKernel::multiply<thrust::complex<float> >(dim3, dim3,
    cudaStream_t, const thrust::complex<float>*, const mpFlow::dtype::index*,
    const thrust::complex<float>*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, thrust::complex<float>*);
template void mpFlow::numeric::sparseMatrixKernel::multiply<thrust::complex<double> >(dim3, dim3,
    cudaStream_t, const thrust::complex<double>*, const mpFlow::dtype::index*,
    const thrust::complex<double>*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, thrust::complex<double>*);
template void mpFlow::numeric::sparseMatrixKernel::multiply<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    const mpFlow::dtype::index*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, mpFlow::dtype::index*);
template void mpFlow::numeric::sparseMatrixKernel::multiply<int>(dim3, dim3,
    cudaStream_t, const int*, const mpFlow::dtype::index*,
    const int*, mpFlow::dtype::size, mpFlow::dtype::size,
    mpFlow::dtype::size, mpFlow::dtype::size, int*);
