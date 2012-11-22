// liblinalgcu
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "../include/linalgcu.h"

// create new sparse matrix
extern "C"
linalgcuError_t linalgcu_sparse_matrix_create(linalgcuSparseMatrix_t* matrixPointer,
    linalgcuMatrix_t matrix, cudaStream_t stream) {
    // check input
    if ((matrixPointer == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init matrix pointer
    *matrixPointer = NULL;

    // create empty sparse matrix
    linalgcuSparseMatrix_t sparseMatrix;
    error = linalgcu_sparse_matrix_create_empty(&sparseMatrix, matrix->rows, matrix->columns,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // convert to sparse_matrix
    error = linalgcu_sparse_matrix_convert(sparseMatrix, matrix, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        linalgcu_sparse_matrix_release(&sparseMatrix);

        return LINALGCU_ERROR;
    }

    // set matrix pointer
    *matrixPointer = sparseMatrix;

    return LINALGCU_SUCCESS;
}

// create empty sparse matrix
extern "C"
linalgcuError_t linalgcu_sparse_matrix_create_empty(linalgcuSparseMatrix_t* matrixPointer,
    linalgcuSize_t rows, linalgcuSize_t columns, cudaStream_t stream) {
    // check input
    if ((matrixPointer == NULL) || (rows == 0) || (columns == 0)) {
        return LINALGCU_ERROR;
    }

    // init matrix pointer
    *matrixPointer = NULL;

    // create struct
    linalgcuSparseMatrix_t sparseMatrix = (linalgcuSparseMatrix_t)malloc(
        sizeof(linalgcuSparseMatrix_s));

    // check success
    if (sparseMatrix == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    sparseMatrix->rows = rows;
    sparseMatrix->columns = columns;
    sparseMatrix->density = 0;
    sparseMatrix->values = NULL;
    sparseMatrix->columnIds = NULL;

    // correct size to block size
    if ((sparseMatrix->rows % LINALGCU_BLOCK_SIZE != 0) && (sparseMatrix->rows != 1)) {
        sparseMatrix->rows = (sparseMatrix->rows / LINALGCU_BLOCK_SIZE + 1) *
            LINALGCU_BLOCK_SIZE;
    }
    if ((sparseMatrix->columns % LINALGCU_BLOCK_SIZE != 0) && (sparseMatrix->columns != 1)) {
        sparseMatrix->columns = (sparseMatrix->columns / LINALGCU_BLOCK_SIZE + 1) *
            LINALGCU_BLOCK_SIZE;
    }

    // create matrices
    if (cudaMalloc((void**)&sparseMatrix->values, sizeof(linalgcuMatrixData_t) *
        sparseMatrix->rows * LINALGCU_SPARSE_SIZE) != cudaSuccess) {
        // cleanup
        linalgcu_sparse_matrix_release(&sparseMatrix);

        return LINALGCU_ERROR;
    }

    if (cudaMalloc((void**)&sparseMatrix->columnIds, sizeof(linalgcuColumnId_t) *
        sparseMatrix->rows * LINALGCU_SPARSE_SIZE) != cudaSuccess) {
        // cleanup
        linalgcu_sparse_matrix_release(&sparseMatrix);

        return LINALGCU_ERROR;
    }

    // set matrix pointer
    *matrixPointer = sparseMatrix;

    return LINALGCU_SUCCESS;
}

// release sparse matrix
extern "C"
linalgcuError_t linalgcu_sparse_matrix_release(linalgcuSparseMatrix_t* matrixPointer) {
    // check input
    if ((matrixPointer == NULL) || (*matrixPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get matrix
    linalgcuSparseMatrix_t matrix = *matrixPointer;

    // release matrices
    if (matrix->values != NULL) {
        cudaFree(matrix->values);
    }

    if (matrix->columnIds != NULL) {
        cudaFree(matrix->columnIds);
    }

    // free struct
    free(matrix);

    // set matrix pointer to NULL
    *matrixPointer = NULL;

    return LINALGCU_SUCCESS;
}

// convert to sparse matrix kernel
__global__ void sparse_create_kernel(linalgcuMatrixData_t* values,
    linalgcuColumnId_t* columnIds, linalgcuMatrixData_t* matrix,
    linalgcuMatrixData_t* elementCount, linalgcuSize_t rows, linalgcuSize_t columns) {
    // get id
    linalgcuSize_t i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    linalgcuSize_t count = 0;

    // init values and columnIds
    for (linalgcuSize_t j = 0; j < LINALGCU_SPARSE_SIZE; j++) {
        values[i * LINALGCU_SPARSE_SIZE + j] = 0.0f;
        columnIds[i * LINALGCU_SPARSE_SIZE + j] = -1;
    }

    // search non-zero elements
    linalgcuMatrixData_t element = 0.0f;
    for (linalgcuSize_t j = 0; j < columns; j++) {
        // get element
        element = matrix[i + j * rows];

        // check for non-zero
        if (element != 0.0f) {
            values[i * LINALGCU_SPARSE_SIZE + count] = element;
            columnIds[i * LINALGCU_SPARSE_SIZE + count] = j;

            // increment count
            count++;

            // check count
            if (count >= LINALGCU_SPARSE_SIZE) {
                break;
            }
        }
    }

    // save element count
    elementCount[i] = (linalgcuMatrixData_t)count;
}

// convert to sparse matrix
extern "C"
linalgcuError_t linalgcu_sparse_matrix_convert(linalgcuSparseMatrix_t sparse,
    linalgcuMatrix_t matrix, cudaStream_t stream) {
    // check input
    if ((sparse == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // create elementCount matrix
    linalgcuMatrix_t elementCount, maxCount;
    error  = linalgcu_matrix_create(&elementCount, sparse->rows, 1, stream);
    error |= linalgcu_matrix_create(&maxCount, sparse->rows, 1, stream);

    // execute kernel
    sparse_create_kernel<<<matrix->rows / LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE,
        0, stream>>>(sparse->values, sparse->columnIds, matrix->deviceData,
        elementCount->deviceData, matrix->rows, matrix->columns);

    // get max count
    error |= linalgcu_matrix_max(maxCount, elementCount, maxCount->rows, stream);
    error |= linalgcu_matrix_copy_to_host(maxCount, stream);
    cudaStreamSynchronize(stream);

    // save density
    sparse->density = (linalgcuSize_t)maxCount->hostData[0];

    // cleanup
    linalgcu_matrix_release(&elementCount);
    linalgcu_matrix_release(&maxCount);

    return error;
}

// sparse matrix multiply kernel
__global__ void sparse_multiply_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* values, linalgcuColumnId_t* columnIds,
    linalgcuMatrixData_t* matrix, linalgcuSize_t rows, linalgcuSize_t columns,
    linalgcuSize_t density) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    linalgcuMatrixData_t res = 0.0f;
    linalgcuColumnId_t id = -1;

    // read column ids to local memory
    __shared__ linalgcuColumnId_t columnId[LINALGCU_SPARSE_SIZE * LINALGCU_SPARSE_SIZE];
    __shared__ linalgcuMatrixData_t value[LINALGCU_SPARSE_SIZE * LINALGCU_SPARSE_SIZE];
    columnId[threadIdx.x * LINALGCU_SPARSE_SIZE + threadIdx.y] = row < rows ?
        columnIds[row * LINALGCU_SPARSE_SIZE + threadIdx.y] : -1;
    value[threadIdx.x * LINALGCU_SPARSE_SIZE + threadIdx.y] = row < rows ?
        values[row * LINALGCU_SPARSE_SIZE + threadIdx.y] : 0.0f;
    __syncthreads();

    // check ids
    if ((row >= rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (linalgcuSize_t j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * LINALGCU_SPARSE_SIZE + j];

         res += id != -1 ? matrix[id + column * rows] *
            value[threadIdx.x * LINALGCU_SPARSE_SIZE + j] : 0.0f;
    }

    // set result
    result[row + column * rows] = res;
}

// sparse matrix multiply
extern "C"
linalgcuError_t linalgcu_sparse_matrix_multiply(linalgcuMatrix_t result,
    linalgcuSparseMatrix_t sparse, linalgcuMatrix_t matrix, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (sparse == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((result->rows != sparse->rows) || (sparse->columns != matrix->rows) ||
        (result->columns != matrix->columns)) {
        return LINALGCU_ERROR;
    }

    // kernel dimension
    dim3 global((result->rows + LINALGCU_SPARSE_SIZE - 1) / LINALGCU_SPARSE_SIZE,
        (result->columns + LINALGCU_SPARSE_SIZE - 1) / LINALGCU_SPARSE_SIZE);
    dim3 local(LINALGCU_SPARSE_SIZE, LINALGCU_SPARSE_SIZE);

    // execute kernel
    sparse_multiply_kernel<<<global, local, 0, stream>>>(result->deviceData, sparse->values,
        sparse->columnIds, matrix->deviceData, result->rows, result->columns, sparse->density);

    return LINALGCU_SUCCESS;
}
