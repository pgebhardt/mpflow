// mpFlow
//
// Copyright (C) 20124 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/uwb/model_kernel.h"

// reduce connectivity and elementalResidual matrix
template <
    class type
>
static __global__ void reduceMatrixKernel(const type* intermediate_matrix,
    const mpFlow::dtype::index* column_ids, mpFlow::dtype::size rows,
    mpFlow::dtype::index offset, type* matrix) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get column id
    mpFlow::dtype::index columnId = column_ids[row * mpFlow::numeric::sparseMatrix::block_size + column];

    // check column id
    if (columnId == mpFlow::dtype::invalid_index) {
        return;
    }

    // reduce matrices
    matrix[row + (column + offset * mpFlow::numeric::sparseMatrix::block_size) * rows] =
        intermediate_matrix[row + columnId * rows];
}

// reduce matrix wrapper
template <
    class type
>
void mpFlow::UWB::modelKernel::reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const type* intermediate_matrix, const dtype::index* column_ids, dtype::size rows,
    dtype::index offset, type* matrix) {
    // call cuda kernel
    reduceMatrixKernel<type><<<blocks, threads, 0, stream>>>(intermediate_matrix,
        column_ids, rows, offset, matrix);

    CudaCheckError();
}

// reduce matrix specialisation
template void mpFlow::UWB::modelKernel::reduceMatrix<mpFlow::dtype::real>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::real*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::real*);
template void mpFlow::UWB::modelKernel::reduceMatrix<mpFlow::dtype::index>(dim3, dim3,
    cudaStream_t, const mpFlow::dtype::index*, const mpFlow::dtype::index*,
    mpFlow::dtype::size, mpFlow::dtype::index, mpFlow::dtype::index*);

// update matrix kernel
static __global__ void updateMatrixKernel(const mpFlow::dtype::index* connectivityMatrix,
    const mpFlow::dtype::real* elementalMatrix, const mpFlow::dtype::real* gamma,
    mpFlow::dtype::real sigmaRef, mpFlow::dtype::size rows, mpFlow::dtype::size columns,
    mpFlow::dtype::real* matrix_values) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc residual matrix element
    mpFlow::dtype::real value = 0.0f;
    mpFlow::dtype::index elementId = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index k = 0; k < columns / mpFlow::numeric::sparseMatrix::block_size; ++k) {
        // get element id
        elementId = connectivityMatrix[row +
            (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows];

        value += elementId != mpFlow::dtype::invalid_index ? elementalMatrix[row +
            (column + k * mpFlow::numeric::sparseMatrix::block_size) * rows] *
            sigmaRef * exp10f(gamma[elementId] / 10.0f) : 0.0f;
    }

    // set residual matrix element
    matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] = value;
}

// update matrix kernel wrapper
void mpFlow::UWB::modelKernel::updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const dtype::index* connectivityMatrix, const dtype::real* elementalMatrix,
    const dtype::real* gamma, dtype::real sigma_ref, dtype::size rows, dtype::size columns,
    dtype::real* matrix_values) {
    // call cuda kernel
    updateMatrixKernel<<<blocks, threads, 0, stream>>>(connectivityMatrix, elementalMatrix,
        gamma, sigma_ref, rows, columns, matrix_values);

    CudaCheckError();
}

// update system matrix kernel
static __global__ void updateSystemMatrixKernel(
    const mpFlow::dtype::real* s_matrix_values, const mpFlow::dtype::real* r_matrix_values,
    const mpFlow::dtype::index* s_matrix_column_ids, const mpFlow::dtype::real* z_matrix,
    mpFlow::dtype::size density, mpFlow::dtype::real scalar, mpFlow::dtype::size z_matrix_rows,
    mpFlow::dtype::real* system_matrix_values) {
    // get row
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;

    // update system matrix
    mpFlow::dtype::index column_id = mpFlow::dtype::invalid_index;
    for (mpFlow::dtype::index column = 0; column < density; ++column) {
        // get column id
        column_id = s_matrix_column_ids[row * mpFlow::numeric::sparseMatrix::block_size + column];

        // update system matrix element
        system_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] =
            column_id != mpFlow::dtype::invalid_index ?
            s_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] +
            r_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column] * scalar +
            z_matrix[row + z_matrix_rows * column_id] :
            system_matrix_values[row * mpFlow::numeric::sparseMatrix::block_size + column];
    }
}

// update system matrix kernel wrapper
void mpFlow::UWB::modelKernel::updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
    const mpFlow::dtype::real* sMatrixValues, const mpFlow::dtype::real* rMatrixValues,
    const mpFlow::dtype::index* sMatrixColumnIds, mpFlow::dtype::size density,
    mpFlow::dtype::real scalar, mpFlow::dtype::real* systemMatrixValues) {
    // TODO
/*
    // call cuda kernel
    updateSystemMatrixKernel<<<blocks, threads, 0, stream>>>(
        s_matrix_values, r_matrix_values, s_matrix_column_ids, z_matrix,
        density, scalar, z_matrix_rows, system_matrix_values);
*/
    CudaCheckError();
}
