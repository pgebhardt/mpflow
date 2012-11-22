// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <locale.h>
#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// specialisation
template class fastEIT::Matrix<fastEIT::dtype::real>;
template class fastEIT::Matrix<fastEIT::dtype::index>;

// create new matrix
template<class type>
Matrix<type>::Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : mHostData(NULL), mDeviceData(NULL), mRows(rows), mColumns(columns) {
    // check input
    if (rows == 0) {
        throw invalid_argument("Matrix::Matrix: rows == 0");
    }
    if (columns == 0) {
        throw invalid_argument("Matrix::Matrix: columns == 0");
    }

    // cuda error
    cudaError_t error = cudaSuccess;

    // correct size to block size
    if ((this->rows() % Matrix<type>::blockSize != 0) && (this->rows() != 1)) {
        this->mRows = (this->rows() / Matrix<type>::blockSize + 1) * Matrix<type>::blockSize;
    }
    if ((this->columns() % Matrix<type>::blockSize != 0) && (this->columns() != 1)) {
        this->mColumns = (this->columns() / Matrix<type>::blockSize + 1) * Matrix<type>::blockSize;
    }

    // create matrix host data memory
    error = cudaHostAlloc((void**)&this->mHostData, sizeof(type) *
        this->rows() * this->columns(), cudaHostAllocDefault);

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::Matrix: create host data memory");
    }

    // create matrix device data memory
    error = cudaMalloc((void**)&this->mDeviceData,
        sizeof(type) * this->rows() * this->columns());

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::Matrix: create device data memory");
    }

    // init data with 0.0
    for (dtype::size i = 0; i < this->rows(); i++) {
        for (dtype::size j = 0; j < this->columns(); j++) {
            this->mHostData[i + this->rows() * j] = 0.0;
        }
    }
    this->copyToDevice(stream);
}

// release matrix
template <class type>
Matrix<type>::~Matrix() {
    // free matrix host data
    cudaFree(this->hostData());

    // free matrix device data
    cudaFree(this->deviceData());
}

// copy matrix
template <class type>
void Matrix<type>::copy(Matrix<type>* other, cudaStream_t stream) {
    // check input
    if (other == NULL) {
        throw invalid_argument("Matrix::copy: other == NULL");
    }

    // check size
    if ((other->rows() != this->rows()) || (other->columns() != this->columns())) {
        throw invalid_argument("Matrix::copy: size");
    }

    // error
    cudaError_t error = cudaSuccess;

    // copy data
    error = cudaMemcpyAsync(other->deviceData(), this->deviceData(),
        sizeof(type) * this->rows() * this->columns(),
        cudaMemcpyDeviceToDevice, stream);

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copy: copy data");
    }
}

// copy to device
template <class type>
void Matrix<type>::copyToDevice(cudaStream_t stream) {
    // check input
    // error
    cudaError_t error = cudaSuccess;

    // copy host buffer to device
    error = cudaMemcpyAsync(this->deviceData(), this->hostData(),
        sizeof(type) * this->rows() * this->columns(),
        cudaMemcpyHostToDevice, stream);

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy data");
    }
}

// copy to host
template <class type>
void Matrix<type>::copyToHost(cudaStream_t stream) {
    // error
    cudaError_t error = cudaSuccess;

    // copy host buffer to device
    error = cudaMemcpyAsync(this->hostData(), this->deviceData(),
        sizeof(type) * this->rows() * this->columns(),
        cudaMemcpyDeviceToHost, stream);

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToHost: copy data");
    }
}

/*
// get matrix element
extern "C"
linalgcuError_t linalgcu_matrix_get_element(linalgcuMatrix_t matrix,
    linalgcuMatrixData_t* value, linalgcuSize_t i, linalgcuSize_t j) {
    // check input
    if ((matrix == NULL) || (value == NULL) ||
        (i >= matrix->rows) || (j >= matrix->columns)) {
        return LINALGCU_ERROR;
    }

    // get value
    *value = matrix->hostData[i + j * matrix->rows];

    return LINALGCU_SUCCESS;
}

// set matrix element
extern "C"
linalgcuError_t linalgcu_matrix_set_element(linalgcuMatrix_t matrix,
    linalgcuMatrixData_t value, linalgcuSize_t i, linalgcuSize_t j) {
    // check input
    if ((matrix == NULL) || (i >= matrix->rows) || (j >= matrix->columns)) {
        return LINALGCU_ERROR;
    }

    // set value
    matrix->hostData[i + j * matrix->rows] = value;

    return LINALGCU_SUCCESS;
}

// create unity matrix
extern "C"
linalgcuError_t linalgcu_matrix_unity(linalgcuMatrix_t* matrixPointer, linalgcuSize_t size,
    cudaStream_t stream) {
    // check input
    if ((matrixPointer == NULL) || (size == 0)) {
        return LINALGCU_ERROR;
    }

    // init matrix pointer
    *matrixPointer = NULL;

    // error
    linalgcuError_t error = LINALGCU_ERROR;

    // create square matrix
    linalgcuMatrix_t matrix = NULL;
    error = linalgcu_matrix_create(&matrix, size, size, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // set matrix elements
    for (linalgcuSize_t i = 0; i < size; i++) {
        linalgcu_matrix_set_element(matrix, 1.0f, i, i);
    }

    // copy to device
    linalgcu_matrix_copy_to_device(matrix, stream);

    // set matrix pointer
    *matrixPointer = matrix;

    return LINALGCU_SUCCESS;
}

// diagonal kernel
__global__ void diagonal_kernel(linalgcuMatrixData_t* result, linalgcuMatrixData_t* matrix,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // set matrix
    result[row + column * rows] = row == column ? matrix[row + column * rows] : 0.0f;
}

// diagonal matrix
LINALGCU_EXTERN_C
linalgcuError_t linalgcu_matrix_diagonal(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
    cudaStream_t stream) {
    // check input
    if ((result == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((result->rows != matrix->rows) || (result->columns != matrix->columns) ||
        (result->rows != result->columns)) {
        return LINALGCU_ERROR;
    }

    // transpose matrix
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE, matrix->columns / LINALGCU_BLOCK_SIZE);

    diagonal_kernel<<<blocks, threads, 0, stream>>>(result->deviceData, matrix->deviceData,
        matrix->rows);

    return LINALGCU_SUCCESS;
}

// transpose kernel
__global__ void transpose_kernel(linalgcuMatrixData_t* result, linalgcuMatrixData_t* matrix,
    linalgcuSize_t rows, linalgcuSize_t columns) {
    // get ids
    linalgcuSize_t i = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t j = blockIdx.y * blockDim.y + threadIdx.y;

    // transpose
    result[j + i * columns] = matrix[i + j * rows];
}

// transpose matrix
extern "C"
linalgcuError_t linalgcu_matrix_transpose(linalgcuMatrix_t result, linalgcuMatrix_t matrix,
    cudaStream_t stream) {
    // check input
    if ((result == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((result->rows != matrix->columns) || (result->columns != matrix->rows)) {
        return LINALGCU_ERROR;
    }

    // transpose matrix
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);
    dim3 blocks(matrix->rows / LINALGCU_BLOCK_SIZE, matrix->columns / LINALGCU_BLOCK_SIZE);

    transpose_kernel<<<blocks, threads, 0, stream>>>(result->deviceData, matrix->deviceData,
        matrix->rows, matrix->columns);

    return LINALGCU_SUCCESS;
}

// add kernel
__global__ void add_kernel(linalgcuMatrixData_t* A, linalgcuMatrixData_t* B,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    A[row + column * rows] += B[row + column * rows];
}

// add matrix
extern "C"
linalgcuError_t linalgcu_matrix_add(linalgcuMatrix_t A, linalgcuMatrix_t B,
    cudaStream_t stream) {
    // check input
    if ((A == NULL) || (B == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((A->rows != B->rows) || (A->columns != B->columns)) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(A->rows == 1 ? 1 : A->rows / LINALGCU_BLOCK_SIZE,
        A->columns == 1 ? 1 : A->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(A->rows == 1 ? 1 : LINALGCU_BLOCK_SIZE,
        A->columns == 1 ? 1 : LINALGCU_BLOCK_SIZE);

    // call kernel
    add_kernel<<<blocks, threads, 0, stream>>>(A->deviceData, B->deviceData, A->rows);

    return LINALGCU_SUCCESS;
}

// matrix multiply
extern "C"
linalgcuError_t linalgcu_matrix_multiply(linalgcuMatrix_t result, linalgcuMatrix_t A,
    linalgcuMatrix_t B, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (A == NULL) || (B == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((A->columns != B->rows) || (result->rows != A->rows) ||
        (result->columns != B->columns)) {
        return LINALGCU_ERROR;
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // multiply matrices
    linalgcuMatrixData_t alpha = 1.0f;
    linalgcuMatrixData_t beta = 0.0f;

    if (B->columns == 1) {
        if (cublasSgemv(handle, CUBLAS_OP_N, A->rows, A->columns, &alpha, A->deviceData,
            A->rows, B->deviceData, 1, &beta, result->deviceData, 1)
            != CUBLAS_STATUS_SUCCESS) {
            return LINALGCU_ERROR;
        }
    }
    else {
        if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->rows, B->columns, A->columns,
            &alpha, A->deviceData, A->rows, B->deviceData, B->rows, &beta,
            result->deviceData, result->rows) != CUBLAS_STATUS_SUCCESS) {
            return LINALGCU_ERROR;
        }
    }

    return LINALGCU_SUCCESS;
}

// scale kernel
__global__ void scale_kernel(linalgcuMatrixData_t* matrix, linalgcuMatrixData_t scalar,
    linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // scale matrix with scalar
    matrix[row + column * rows] *= scalar;
}

// scalar multiply matrix
extern "C"
linalgcuError_t linalgcu_matrix_scalar_multiply(linalgcuMatrix_t matrix,
    linalgcuMatrixData_t scalar, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        return LINALGCU_ERROR;
    }

    // dimension
    dim3 blocks(matrix->rows == 1 ? 1 : matrix->rows / LINALGCU_BLOCK_SIZE,
        matrix->columns == 1 ? 1 : matrix->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(matrix->rows == 1 ? 1 : LINALGCU_BLOCK_SIZE,
        matrix->columns == 1 ? 1 : LINALGCU_BLOCK_SIZE);

    // call kernel
    scale_kernel<<<blocks, threads, 0, stream>>>(matrix->deviceData, scalar, matrix->rows);

    return LINALGCU_SUCCESS;
}

// vector dot product kernel
__global__ void vector_dot_product_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* a, linalgcuMatrixData_t* b, linalgcuSize_t rows) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

// vector dot product
extern "C"
linalgcuError_t linalgcu_matrix_vector_dot_product(linalgcuMatrix_t result,
    linalgcuMatrix_t a, linalgcuMatrix_t b, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (a == NULL) || (b == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((result->rows != a->rows) || (result->rows != b->rows)) {
        return LINALGCU_ERROR;
    }

    // get minimum colums
    linalgcuSize_t columns = min(min(result->columns, a->columns), b->columns);

    // kernel dimension
    dim3 global(result->rows / LINALGCU_BLOCK_SIZE, columns == 1 ? 1 : columns / LINALGCU_BLOCK_SIZE);
    dim3 local(LINALGCU_BLOCK_SIZE, columns == 1 ? 1 : LINALGCU_BLOCK_SIZE);

    // call dot kernel
    vector_dot_product_kernel<<<global, local, 0, stream>>>(result->deviceData, a->deviceData, b->deviceData,
        result->rows);

    // sum
    return linalgcu_matrix_sum(result, result, stream);
}

// sum kernel
__global__ void sum_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* vector, linalgcuSize_t rows, linalgcuSize_t offset) {
    // get column
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    linalgcuSize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ linalgcuMatrixData_t res[LINALGCU_BLOCK_SIZE * LINALGCU_BLOCK_SIZE];
    res[lid + threadIdx.y * LINALGCU_BLOCK_SIZE] = gid * offset < rows ? vector[gid * offset + column * rows] : 0.0f;

    // reduce
    res[lid + threadIdx.y * LINALGCU_BLOCK_SIZE] += (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * LINALGCU_BLOCK_SIZE] : 0.0f;
    res[lid + threadIdx.y * LINALGCU_BLOCK_SIZE] += (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * LINALGCU_BLOCK_SIZE] : 0.0f;
    res[lid + threadIdx.y * LINALGCU_BLOCK_SIZE] += (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * LINALGCU_BLOCK_SIZE] : 0.0f;
    res[lid + threadIdx.y * LINALGCU_BLOCK_SIZE] += (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * LINALGCU_BLOCK_SIZE] : 0.0f;
    __syncthreads();

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows] = res[lid + threadIdx.y * LINALGCU_BLOCK_SIZE];
}

// sum
extern "C"
linalgcuError_t linalgcu_matrix_sum(linalgcuMatrix_t result, linalgcuMatrix_t vector,
    cudaStream_t stream) {
    // check success
    if ((result == NULL) || (vector == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if (result->rows != vector->rows) {
        return LINALGCU_ERROR;
    }

    // get minimum columns
    linalgcuSize_t columns = min(result->columns, vector->columns);

    // kernel settings
    dim3 global(result->rows / LINALGCU_BLOCK_SIZE, columns == 1 ? 1 : columns / LINALGCU_BLOCK_SIZE);
    dim3 local(LINALGCU_BLOCK_SIZE, columns == 1 ? 1 : LINALGCU_BLOCK_SIZE);
    linalgcuSize_t offset = 1;

    // start kernel once
    sum_kernel<<<global, local, 0, stream>>>(
        result->deviceData, vector->deviceData, result->rows, offset);

    // start kernel
    do {
        // update settings
        offset *= LINALGCU_BLOCK_SIZE;
        global.x = (global.x + LINALGCU_BLOCK_SIZE - 1) /  LINALGCU_BLOCK_SIZE;

        sum_kernel<<<global, local, 0, stream>>>(
            result->deviceData, result->deviceData, result->rows, offset);

    }
    while (offset * LINALGCU_BLOCK_SIZE < result->rows);

    return LINALGCU_SUCCESS;
}

// min kernel
__global__ void min_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* vector, linalgcuSize_t rows, linalgcuSize_t maxIndex,
    linalgcuSize_t offset) {
    // get id
    linalgcuSize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ linalgcuMatrixData_t res[LINALGCU_BLOCK_SIZE];
    res[lid] = gid * offset < maxIndex ? vector[gid * offset] : NAN;

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

// min
extern "C"
linalgcuError_t linalgcu_matrix_min(linalgcuMatrix_t result, linalgcuMatrix_t vector,
    linalgcuSize_t maxIndex, cudaStream_t stream) {
    // check success
    if ((result == NULL) || (vector == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if (result->rows != vector->rows) {
        return LINALGCU_ERROR;
    }

    // kernel settings
    linalgcuSize_t global = result->rows / LINALGCU_BLOCK_SIZE;
    linalgcuSize_t offset = 1;

    // start kernel once
    min_kernel<<<global, LINALGCU_BLOCK_SIZE, 0, stream>>>(
        result->deviceData, vector->deviceData, result->rows, maxIndex,
        offset);

    // start kernel
    do {
        // update settings
        offset *= LINALGCU_BLOCK_SIZE;
        global = (global + LINALGCU_BLOCK_SIZE - 1) / LINALGCU_BLOCK_SIZE;

        min_kernel<<<global, LINALGCU_BLOCK_SIZE, 0, stream>>>(
            result->deviceData, result->deviceData, result->rows, maxIndex,
            offset);

    }
    while (offset * LINALGCU_BLOCK_SIZE < result->rows);

    return LINALGCU_SUCCESS;
}

// max kernel
__global__ void max_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* vector, linalgcuSize_t rows, linalgcuSize_t maxIndex,
    linalgcuSize_t offset) {
    // get id
    linalgcuSize_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ linalgcuMatrixData_t res[LINALGCU_BLOCK_SIZE];
    res[lid] = gid * offset < maxIndex ? vector[gid * offset] : NAN;

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

// max
extern "C"
linalgcuError_t linalgcu_matrix_max(linalgcuMatrix_t result, linalgcuMatrix_t vector,
    linalgcuSize_t maxIndex, cudaStream_t stream) {
    // check success
    if ((result == NULL) || (vector == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if (result->rows != vector->rows) {
        return LINALGCU_ERROR;
    }

    // kernel settings
    linalgcuSize_t global = result->rows / LINALGCU_BLOCK_SIZE;
    linalgcuSize_t offset = 1;

    // start kernel once
    max_kernel<<<global, LINALGCU_BLOCK_SIZE, 0, stream>>>(
        result->deviceData, vector->deviceData, result->rows, maxIndex,
        offset);

    // start kernel
    do {
        // update settings
        offset *= LINALGCU_BLOCK_SIZE;
        global = (global + LINALGCU_BLOCK_SIZE - 1) / LINALGCU_BLOCK_SIZE;

        max_kernel<<<global, LINALGCU_BLOCK_SIZE, 0, stream>>>(
            result->deviceData, result->deviceData, result->rows, maxIndex,
            offset);

    }
    while (offset * LINALGCU_BLOCK_SIZE < result->rows);

    return LINALGCU_SUCCESS;
}

// save matrix to file
extern "C"
linalgcuError_t linalgcu_matrix_save(const char* fileName, linalgcuMatrix_t matrix) {
    // check input
    if ((fileName == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // open file
    FILE* file = fopen(fileName, "w");

    // check success
    if (file == NULL) {
        return LINALGCU_ERROR;
    }

    // set local
    setlocale(LC_NUMERIC, "C");

    // write matrix
    for (linalgcuSize_t i = 0; i < matrix->rows; i++) {
        for (linalgcuSize_t j = 0; j < matrix->columns - 1; j++) {
            // write single element
            fprintf(file, "%f ", matrix->hostData[i + matrix->rows * j]);
        }

        // write last row element
        fprintf(file, "%f\n", matrix->hostData[i + (matrix->columns - 1) * matrix->rows]);
    }

    // cleanup
    fclose(file);

    return LINALGCU_SUCCESS;
}

// load matrix from file
extern "C"
linalgcuError_t linalgcu_matrix_load(linalgcuMatrix_t* resultPointer, const char* fileName,
    cudaStream_t stream) {
    // check input
    if ((fileName == NULL) || (resultPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // init result ointer
    *resultPointer = NULL;

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // open file
    FILE* file = fopen(fileName, "r");

    // check success
    if (file == NULL) {
        return LINALGCU_ERROR;
    }

    // set local
    setlocale(LC_NUMERIC, "C");

    // line buffer
    char buffer[10240];

    // get size of matrix
    linalgcuSize_t rows = 0;
    linalgcuSize_t columns = 0;

    // get x size
    while (fgets(buffer, 10240, file) != NULL) {
        rows++;
    }
    fseek(file, 0, SEEK_SET);

    // check x size
    if (rows == 0) {
        // cleanup
        fclose(file);

        return LINALGCU_ERROR;
    }

    // get first line
    fgets(buffer, 10240, file);

    // get y size
    if (strtok(buffer, " \n")) {
        columns++;

        while (strtok(NULL, " ") != NULL) {
            columns++;
        }
    }
    fseek(file, 0, SEEK_SET);

    // check y size
    if (columns == 0) {
        // cleanup
        fclose(file);

        return LINALGCU_ERROR;
    }

    // create matrix
    linalgcuMatrix_t matrix = NULL;
    error = linalgcu_matrix_create(&matrix, rows, columns, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        // cleanup
        fclose(file);

        return error;
    }

    // read data
    char* element = NULL;
    for (linalgcuSize_t i = 0; i < rows; i++) {
        // read line
        fgets(buffer, 10240, file);

        // read first element
        element = strtok(buffer, " \n");

        if (element == NULL) {
            // cleanup
            linalgcu_matrix_release(&matrix);
            fclose(file);

            return LINALGCU_ERROR;
        }

        linalgcu_matrix_set_element(matrix, atof(element), i, 0);

        for (linalgcuSize_t j = 1; j < columns; j++) {
            // get element
            element = strtok(NULL, " ");

            // set element
            matrix->hostData[i + matrix->rows * j] = atof(element);
        }
    }

    // copy to device
    linalgcu_matrix_copy_to_device(matrix, stream);

    // set result pointer
    *resultPointer = matrix;

    return LINALGCU_SUCCESS;
}*/
