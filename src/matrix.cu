// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create new matrix
template<class type>
Matrix<type>::Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : mHostData(NULL), mDeviceData(NULL), mDataRows(rows), mDataColumns(columns) {
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
    if ((this->dataRows() % Matrix<type>::blockSize != 0) && (this->dataRows() != 1)) {
        this->mDataRows = (this->dataRows() / Matrix<type>::blockSize + 1) * Matrix<type>::blockSize;
    }
    if ((this->dataColumns() % Matrix<type>::blockSize != 0) && (this->dataColumns() != 1)) {
        this->mDataColumns = (this->dataColumns() / Matrix<type>::blockSize + 1) * Matrix<type>::blockSize;
    }

    // create matrix host data memory
    error = cudaHostAlloc((void**)&this->mHostData, sizeof(type) *
        this->dataRows() * this->dataColumns(), cudaHostAllocDefault);

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::Matrix: create host data memory");
    }

    // create matrix device data memory
    error = cudaMalloc((void**)&this->mDeviceData,
        sizeof(type) * this->dataRows() * this->dataColumns());

    // check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::Matrix: create device data memory");
    }

    // init data with 0.0
    for (dtype::size i = 0; i < this->dataRows(); i++) {
        for (dtype::size j = 0; j < this->dataColumns(); j++) {
            this->mHostData[i + this->dataRows() * j] = 0.0;
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
    if ((other->dataRows() != this->dataRows()) || (other->dataColumns() != this->dataColumns())) {
        throw invalid_argument("Matrix::copy: size");
    }

    // copy data
    cudaMemcpyAsync(this->deviceData(), other->deviceData(),
        sizeof(type) * this->dataRows() * this->dataColumns(),
        cudaMemcpyDeviceToDevice, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// copy to device
template <class type>
void Matrix<type>::copyToDevice(cudaStream_t stream) {
    // copy host buffer to device
    cudaMemcpyAsync(this->deviceData(), this->hostData(),
        sizeof(type) * this->dataRows() * this->dataColumns(),
        cudaMemcpyHostToDevice, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// copy to host
template <class type>
void Matrix<type>::copyToHost(cudaStream_t stream) {
    // copy host buffer to device
    cudaMemcpyAsync(this->hostData(), this->deviceData(),
        sizeof(type) * this->dataRows() * this->dataColumns(),
        cudaMemcpyDeviceToHost, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// add kernel
template<class type>
__global__ void addKernel(type* A, type* B,
    dtype::size rows) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    A[row + column * rows] += B[row + column * rows];
}

// add matrix
template <class type>
void Matrix<type>::add(Matrix<type>* value, cudaStream_t stream) {
    // check input
    if (value == NULL) {
        throw invalid_argument("Matrix::add: other == NULL");
    }

    // check size
    if ((this->dataRows() != value->dataRows()) || (this->dataColumns() != value->dataColumns())) {
        throw invalid_argument("Matrix::add: size");
    }

    // dimension
    dim3 blocks(this->dataRows() == 1 ? 1 : this->dataRows() / Matrix<type>::blockSize,
        this->dataColumns() == 1 ? 1 : this->dataColumns() / Matrix<type>::blockSize);
    dim3 threads(this->dataRows() == 1 ? 1 : Matrix<type>::blockSize,
        this->dataColumns() == 1 ? 1 : Matrix<type>::blockSize);

    // call kernel
    addKernel<type><<<blocks, threads, 0, stream>>>(this->deviceData(), value->deviceData(),
        this->dataRows());
}

// matrix multiply
template <>
void Matrix<dtype::real>::multiply(Matrix<dtype::real>* A, Matrix<dtype::real>* B, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (A == NULL) {
        throw invalid_argument("Matrix::multiply: A == NULL");
    }
    if (B == NULL) {
        throw invalid_argument("Matrix::multiply: B == NULL");
    }

    // check size
    if ((A->dataColumns() != B->dataRows()) || (this->dataRows() != A->dataRows()) ||
        (this->dataColumns() != B->dataColumns())) {
        throw invalid_argument("Matrix::multiply: size");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // multiply matrices
    dtype::real alpha = 1.0f;
    dtype::real beta = 0.0f;

    if (B->dataColumns() == 1) {
        if (cublasSgemv(handle, CUBLAS_OP_N, A->dataRows(), A->dataColumns(), &alpha, A->deviceData(),
            A->dataRows(), B->deviceData(), 1, &beta, this->deviceData(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw logic_error("Matrix::multiply: cublasSgemv");
        }
    }
    else {
        if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->dataRows(), B->dataColumns(), A->dataColumns(),
            &alpha, A->deviceData(), A->dataRows(), B->deviceData(), B->dataRows(), &beta,
            this->deviceData(), this->dataRows()) != CUBLAS_STATUS_SUCCESS) {
            throw logic_error("Matrix::multiply: cublasSgemm");
        }
    }
}
// matrix multiply
template <class type>
void Matrix<type>::multiply(Matrix<type>* A, Matrix<type>* B, cublasHandle_t handle, cudaStream_t stream) {
    throw logic_error("Matrix::multiply: not supported dtype");
}

// scale kernel
template <class type>
__global__ void scale_kernel(type* matrix, type scalar, dtype::size rows) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // scale matrix with scalar
    matrix[row + column * rows] *= scalar;
}

// scalar multiply matrix
template <class type>
void Matrix<type>::scalarMultiply(type scalar, cudaStream_t stream) {
    // dimension
    dim3 blocks(this->dataRows() == 1 ? 1 : this->dataRows() / Matrix<type>::blockSize,
        this->dataColumns() == 1 ? 1 : this->dataColumns() / Matrix<type>::blockSize);
    dim3 threads(this->dataRows() == 1 ? 1 : Matrix<type>::blockSize,
        this->dataColumns() == 1 ? 1 : Matrix<type>::blockSize);

    // call kernel
    scale_kernel<type><<<blocks, threads, 0, stream>>>(this->deviceData(), scalar, this->dataRows());
}

// vector dot product kernel
template <class type>
__global__ void vectorDotProductKernel(type* result, type* a, type* b,
    dtype::size rows) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

// vector dot product
template <class type>
void Matrix<type>::vectorDotProduct(Matrix<type>* A, Matrix<type>* B, cudaStream_t stream) {
    // check input
    if (A == NULL) {
        throw invalid_argument("Matrix::vectorDotProduct: A == NULL");
    }
    if (B == NULL) {
        throw invalid_argument("Matrix::vectorDotProduct: B == NULL");
    }

    // check size
    if ((this->dataRows() != A->dataRows()) || (this->dataRows() != B->dataRows())) {
        throw invalid_argument("Matrix::vectorDotProduct: size");
    }

    // get minimum colums
    dtype::size columns = std::min(std::min(this->dataColumns(), A->dataColumns()), B->dataColumns());

    // kernel dimension
    dim3 global(this->dataRows() / Matrix<type>::blockSize, columns == 1 ? 1 : columns / Matrix<type>::blockSize);
    dim3 local(Matrix<type>::blockSize, columns == 1 ? 1 : Matrix<type>::blockSize);

    // call dot kernel
    vectorDotProductKernel<type><<<global, local, 0, stream>>>(this->deviceData(), A->deviceData(), B->deviceData(),
        this->dataRows());

    // sum
    this->sum(this, stream);
}

// sum kernel
template <class type>
__global__ void sumKernel(type* result, type* vector, dtype::size rows,
    dtype::size offset) {
    // get column
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // get id
    dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[Matrix<type>::blockSize * Matrix<type>::blockSize];
    res[lid + threadIdx.y * Matrix<type>::blockSize] = gid * offset < rows ? vector[gid * offset + column * rows] : 0.0f;

    // reduce
    res[lid + threadIdx.y * Matrix<type>::blockSize] += (lid % 2 == 0) ? res[lid + 1 + threadIdx.y * Matrix<type>::blockSize] : 0.0f;
    res[lid + threadIdx.y * Matrix<type>::blockSize] += (lid % 4 == 0) ? res[lid + 2 + threadIdx.y * Matrix<type>::blockSize] : 0.0f;
    res[lid + threadIdx.y * Matrix<type>::blockSize] += (lid % 8 == 0) ? res[lid + 4 + threadIdx.y * Matrix<type>::blockSize] : 0.0f;
    res[lid + threadIdx.y * Matrix<type>::blockSize] += (lid % 16 == 0) ? res[lid + 8 + threadIdx.y * Matrix<type>::blockSize] : 0.0f;
    __syncthreads();

    // stop rest of worker
    if (lid != 0) {
        return;
    }

    // write to global memory
    result[gid * offset + column * rows] = res[lid + threadIdx.y * Matrix<type>::blockSize];
}

// sum
template <class type>
void Matrix<type>::sum(Matrix<type>* value, cudaStream_t stream) {
    // check input
    if (value == NULL) {
        throw invalid_argument("Matrix::sum: value == NULL");
    }

    // check size
    if (this->dataRows() != value->dataRows()) {
        throw invalid_argument("Matrix::sum: size");
    }

    // get minimum columns
    dtype::size columns = std::min(this->dataColumns(), value->dataColumns());

    // kernel settings
    dim3 global(this->dataRows() / Matrix<type>::blockSize, columns == 1 ? 1 : columns / Matrix<type>::blockSize);
    dim3 local(Matrix<type>::blockSize, columns == 1 ? 1 : Matrix<type>::blockSize);
    dtype::size offset = 1;

    // start kernel once
    sumKernel<type><<<global, local, 0, stream>>>(
        this->deviceData(), value->deviceData(), this->dataRows(), offset);

    // start kernel
    do {
        // update settings
        offset *= Matrix<type>::blockSize;
        global.x = (global.x + Matrix<type>::blockSize - 1) /  Matrix<type>::blockSize;

        sumKernel<<<global, local, 0, stream>>>(
            this->deviceData(), this->deviceData(), this->dataRows(), offset);

    }
    while (offset * Matrix<type>::blockSize < this->dataRows());
}

// min kernel
template <class type>
__global__ void minKernel(type* result, type* vector, dtype::size rows, dtype::size maxIndex,
    dtype::size offset) {
    // get id
    dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[Matrix<type>::blockSize];
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
template <class type>
void Matrix<type>::min(Matrix<type>* value, dtype::size maxIndex, cudaStream_t stream) {
    // check input
    if (value == NULL) {
        throw invalid_argument("Matrix::min: value == NULL");
    }

    // check size
    if (this->dataRows() != value->dataRows()) {
        throw invalid_argument("Matrix::min: size");
    }

    // kernel settings
    dtype::size global = this->dataRows() / Matrix<type>::blockSize;
    dtype::size offset = 1;

    // start kernel once
    minKernel<type><<<global, Matrix<type>::blockSize, 0, stream>>>(
        this->deviceData(), value->deviceData(), this->dataRows(), maxIndex,
        offset);

    // start kernel
    do {
        // update settings
        offset *= Matrix<type>::blockSize;
        global = (global + Matrix<type>::blockSize - 1) / Matrix<type>::blockSize;

        minKernel<type><<<global, Matrix<type>::blockSize, 0, stream>>>(
            this->deviceData(), this->deviceData(), this->dataRows(), maxIndex,
            offset);

    }
    while (offset * Matrix<type>::blockSize < this->dataRows());
}

// max kernel
template <class type>
__global__ void maxKernel(type* result, type* vector, dtype::size rows, dtype::size maxIndex,
    dtype::size offset) {
    // get id
    dtype::index gid = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index lid = threadIdx.x;

    // copy data to shared memory
    __volatile __shared__ type res[Matrix<type>::blockSize];
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
template <class type>
void Matrix<type>::max(Matrix<type>* value, dtype::size maxIndex, cudaStream_t stream) {
    // check input
    if (value == NULL) {
        throw invalid_argument("Matrix::max: value == NULL");
    }

    // check size
    if (this->dataRows() != value->dataRows()) {
        throw invalid_argument("Matrix::max: size");
    }

    // kernel settings
    dtype::size global = this->dataRows() / Matrix<type>::blockSize;
    dtype::size offset = 1;

    // start kernel once
    maxKernel<type><<<global, Matrix<type>::blockSize, 0, stream>>>(
        this->deviceData(), value->deviceData(), this->dataRows(), maxIndex,
        offset);

    // start kernel
    do {
        // update settings
        offset *= Matrix<type>::blockSize;
        global = (global + Matrix<type>::blockSize - 1) / Matrix<type>::blockSize;

        maxKernel<type><<<global, Matrix<type>::blockSize, 0, stream>>>(
            this->deviceData(), this->deviceData(), this->dataRows(), maxIndex,
            offset);

    }
    while (offset * Matrix<type>::blockSize < this->dataRows());
}
/*
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

// specialisation
template class fastEIT::Matrix<fastEIT::dtype::real>;
template class fastEIT::Matrix<fastEIT::dtype::index>;
