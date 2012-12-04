// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <vector>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"

// create new matrix
template<class type>
fastEIT::Matrix<type>::Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : host_data_(NULL), device_data_(NULL), rows_(rows), data_rows_(rows),
        columns_(columns), data_columns_(columns) {
    // check input
    if (rows == 0) {
        throw std::invalid_argument("Matrix::Matrix: rows == 0");
    }
    if (columns == 0) {
        throw std::invalid_argument("Matrix::Matrix: columns == 0");
    }

    // cuda error
    cudaError_t error = cudaSuccess;

    // correct size to block size
    if ((this->rows() % Matrix<type>::block_size != 0) && (this->rows() != 1)) {
        this->data_rows_ = (this->rows() / Matrix<type>::block_size + 1) * Matrix<type>::block_size;
    }
    if ((this->columns() % Matrix<type>::block_size != 0) && (this->columns() != 1)) {
        this->data_columns_ = (this->columns() / Matrix<type>::block_size + 1) * Matrix<type>::block_size;
    }

    // create matrix host data memory
    error = cudaHostAlloc((void**)&this->host_data_, sizeof(type) *
        this->data_rows() * this->data_columns(), cudaHostAllocDefault);

    // check success
    if (error != cudaSuccess) {
        throw std::logic_error("Matrix::Matrix: create host data memory");
    }

    // create matrix device data memory
    error = cudaMalloc((void**)&this->device_data_,
        sizeof(type) * this->data_rows() * this->data_columns());

    // check success
    if (error != cudaSuccess) {
        throw std::logic_error("Matrix::Matrix: create device data memory");
    }

    // init data with 0.0
    for (dtype::size i = 0; i < this->data_rows(); i++) {
        for (dtype::size j = 0; j < this->data_columns(); j++) {
            this->host_data()[i + this->data_rows() * j] = 0.0;
        }
    }
    this->copyToDevice(stream);
}

// release matrix
template <class type>
fastEIT::Matrix<type>::~Matrix() {
    // free matrix host data
    cudaFree(this->host_data_);

    // free matrix device data
    cudaFree(this->device_data_);
}

// copy matrix
template <class type>
void fastEIT::Matrix<type>::copy(const Matrix<type>& other, cudaStream_t stream) {
    // check size
    if ((other.data_rows() != this->data_rows()) || (other.data_columns() != this->data_columns())) {
        throw std::invalid_argument("Matrix::copy: size");
    }

    // copy data
    cudaMemcpyAsync(this->device_data(), other.device_data(),
        sizeof(type) * this->data_rows() * this->data_columns(),
        cudaMemcpyDeviceToDevice, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// copy to device
template <class type>
void fastEIT::Matrix<type>::copyToDevice(cudaStream_t stream) {
    // copy host buffer to device
    cudaMemcpyAsync(this->device_data(), this->host_data(),
        sizeof(type) * this->data_rows() * this->data_columns(),
        cudaMemcpyHostToDevice, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// copy to host
template <class type>
void fastEIT::Matrix<type>::copyToHost(cudaStream_t stream) {
    // copy host buffer to device
    cudaMemcpyAsync(this->host_data(), this->device_data(),
        sizeof(type) * this->data_rows() * this->data_columns(),
        cudaMemcpyDeviceToHost, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// add kernel
template<class type>
__global__ void addKernel(const type* matrix, fastEIT::dtype::size rows, type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // add B to A
    result[row + column * rows] += matrix[row + column * rows];
}

// add matrix
template <class type>
void fastEIT::Matrix<type>::add(const Matrix<type>& value, cudaStream_t stream) {
    // check size
    if ((this->data_rows() != value.data_rows()) || (this->data_columns() != value.data_columns())) {
        throw std::invalid_argument("Matrix::add: size");
    }

    // dimension
    dim3 blocks(this->data_rows() == 1 ? 1 : this->data_rows() / Matrix<type>::block_size,
        this->data_columns() == 1 ? 1 : this->data_columns() / Matrix<type>::block_size);
    dim3 threads(this->data_rows() == 1 ? 1 : Matrix<type>::block_size,
        this->data_columns() == 1 ? 1 : Matrix<type>::block_size);

    // call kernel
    addKernel<type><<<blocks, threads, 0, stream>>>(value.device_data(), this->data_rows(), this->device_data());
}


// matrix multiply
template <class type>
void fastEIT::Matrix<type>::multiply(const Matrix<type>& A, const Matrix<type>& B,
    cublasHandle_t handle, cudaStream_t stream) {
    throw std::logic_error("Matrix::multiply: not supported dtype");
}

// specialisation for dtype::real
namespace fastEIT {
    template <>
    void Matrix<dtype::real>::multiply(const Matrix<dtype::real>& A,
        const Matrix<dtype::real>& B, cublasHandle_t handle, cudaStream_t stream) {
        // check input
        if (handle == NULL) {
            throw std::invalid_argument("Matrix::multiply: handle == NULL");
        }

        // check size
        if ((A.data_columns() != B.data_rows()) || (this->data_rows() != A.data_rows()) ||
            (this->data_columns() != B.data_columns())) {
            throw std::invalid_argument("Matrix::multiply: size");
        }

        // set cublas stream
        cublasSetStream(handle, stream);

        // multiply matrices
        dtype::real alpha = 1.0f;
        dtype::real beta = 0.0f;

        if (B.data_columns() == 1) {
            if (cublasSgemv(handle, CUBLAS_OP_N, A.data_rows(), A.data_columns(), &alpha, A.device_data(),
                A.data_rows(), B.device_data(), 1, &beta, this->device_data(), 1)
                != CUBLAS_STATUS_SUCCESS) {
                throw std::logic_error("Matrix::multiply: cublasSgemv");
            }
        }
        else {
            if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A.data_rows(), B.data_columns(), A.data_columns(),
                &alpha, A.device_data(), A.data_rows(), B.device_data(), B.data_rows(), &beta,
                this->device_data(), this->data_rows()) != CUBLAS_STATUS_SUCCESS) {
                throw std::logic_error("Matrix::multiply: cublasSgemm");
            }
        }
    }
}

// scale kernel
template <class type>
__global__ void scaleKernel(type scalar, fastEIT::dtype::size rows, type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // scale matrix with scalar
    result[row + column * rows] *= scalar;
}

// scalar multiply matrix
template <class type>
void fastEIT::Matrix<type>::scalarMultiply(type scalar, cudaStream_t stream) {
    // dimension
    dim3 blocks(this->data_rows() == 1 ? 1 : this->data_rows() / Matrix<type>::block_size,
        this->data_columns() == 1 ? 1 : this->data_columns() / Matrix<type>::block_size);
    dim3 threads(this->data_rows() == 1 ? 1 : Matrix<type>::block_size,
        this->data_columns() == 1 ? 1 : Matrix<type>::block_size);

    // call kernel
    scaleKernel<type><<<blocks, threads, 0, stream>>>(scalar, this->data_rows(), this->device_data());
}

// vector dot product kernel
template <class type>
__global__ void vectorDotProductKernel(const type* a, const type* b, fastEIT::dtype::size rows,
    type* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // elementwise multiply
    result[row + column * rows] = a[row + column * rows] * b[row + column * rows];
}

// vector dot product
template <class type>
void fastEIT::Matrix<type>::vectorDotProduct(const Matrix<type>& A, const Matrix<type>& B,
    cudaStream_t stream) {
    // check size
    if ((this->data_rows() != A.data_rows()) || (this->data_rows() != B.data_rows())) {
        throw std::invalid_argument("Matrix::vectorDotProduct: size");
    }

    // get minimum colums
    dtype::size columns = std::min(std::min(this->data_columns(), A.data_columns()), B.data_columns());

    // kernel dimension
    dim3 global(this->data_rows() / Matrix<type>::block_size, columns == 1 ? 1 : columns / Matrix<type>::block_size);
    dim3 local(Matrix<type>::block_size, columns == 1 ? 1 : Matrix<type>::block_size);

    // call dot kernel
    vectorDotProductKernel<type><<<global, local, 0, stream>>>(A.device_data(), B.device_data(), this->data_rows(),
        this->device_data());

    // sum
    this->sum(*this, stream);
}

// sum kernel
template <class type>
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

// sum
template <class type>
void fastEIT::Matrix<type>::sum(const Matrix<type>& value, cudaStream_t stream) {
    // check size
    if (this->data_rows() != value.data_rows()) {
        throw std::invalid_argument("Matrix::sum: size");
    }

    // get minimum columns
    dtype::size columns = std::min(this->data_columns(), value.data_columns());

    // kernel settings
    dim3 global(this->data_rows() / Matrix<type>::block_size, columns == 1 ? 1 : columns / Matrix<type>::block_size);
    dim3 local(Matrix<type>::block_size, columns == 1 ? 1 : Matrix<type>::block_size);
    dtype::size offset = 1;

    // start kernel once
    sumKernel<type><<<global, local, 0, stream>>>(value.device_data(), this->data_rows(),
        offset, this->device_data());

    // start kernel
    do {
        // update settings
        offset *= Matrix<type>::block_size;
        global.x = (global.x + Matrix<type>::block_size - 1) /  Matrix<type>::block_size;

        sumKernel<<<global, local, 0, stream>>>(this->device_data(), this->data_rows(), offset,
            this->device_data());

    }
    while (offset * Matrix<type>::block_size < this->data_rows());
}

// min kernel
template <class type>
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

// min
template <class type>
void fastEIT::Matrix<type>::min(const Matrix<type>& value, cudaStream_t stream) {
    // check size
    if (this->data_rows() != value.data_rows()) {
        throw std::invalid_argument("Matrix::min: size");
    }

    // kernel settings
    dtype::size global = this->data_rows() / Matrix<type>::block_size;
    dtype::size offset = 1;

    // start kernel once
    minKernel<type><<<global, Matrix<type>::block_size, 0, stream>>>(value.device_data(),
        this->rows(), offset, this->device_data());

    // start kernel
    do {
        // update settings
        offset *= Matrix<type>::block_size;
        global = (global + Matrix<type>::block_size - 1) / Matrix<type>::block_size;

        minKernel<type><<<global, Matrix<type>::block_size, 0, stream>>>(this->device_data(),
            this->rows(), offset, this->device_data());

    }
    while (offset * Matrix<type>::block_size < this->data_rows());
}

// max kernel
template <class type>
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
}

// max
template <class type>
void fastEIT::Matrix<type>::max(const Matrix<type>& value, cudaStream_t stream) {
    // check size
    if (this->data_rows() != value.data_rows()) {
        throw std::invalid_argument("Matrix::max: size");
    }

    // kernel settings
    dtype::size global = this->data_rows() / Matrix<type>::block_size;
    dtype::size offset = 1;

    // start kernel once
    maxKernel<type><<<global, Matrix<type>::block_size, 0, stream>>>(value.device_data(),
        this->rows(), offset, this->device_data());

    // start kernel
    do {
        // update settings
        offset *= Matrix<type>::block_size;
        global = (global + Matrix<type>::block_size - 1) / Matrix<type>::block_size;

        maxKernel<type><<<global, Matrix<type>::block_size, 0, stream>>>(this->device_data(),
            this->rows(), offset, this->device_data());

    }
    while (offset * Matrix<type>::block_size < this->data_rows());
}

// load matrix from file
template <
    class type
>
fastEIT::Matrix<type>* fastEIT::matrix::loadtxt(const std::string filename, cudaStream_t stream) {
    // open file stream
    std::ifstream file;
    file.open(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("matrix::loadtxt: cannot open file!");
    }

    // read lines
    std::vector<std::string> lines;
    std::string line;
    while (!file.eof()) {
        // read line
        getline(file, line);

        // add line to vector
        lines.push_back(line);
    }

    // close file
    file.close();

    // read values
    std::vector<std::vector<type> > values;
    for (std::vector<std::string>::iterator line = lines.begin();
        line != lines.end(); ++line) {
        // create row vector
        std::vector<type> row;
        type value;

        // create string stream
        std::stringstream line_stream(*line);

        // read values of line
        while (!line_stream.eof()) {
            line_stream >> value;
            row.push_back(value);
        }

        // add row
        values.push_back(row);
    }

    // create matrix
    Matrix<type>* matrix = new Matrix<type>(values.size() - 1, values[0].size(), stream);

    // add values
    for (dtype::index row = 0; row < matrix->rows(); ++row) {
        for (dtype::index column = 0; column < matrix->columns(); ++column) {
            (*matrix)(row, column) = values[row][column];
        }
    }

    // copy data to device
    matrix->copyToDevice(stream);

    return matrix;
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
*/

// specialisation
template fastEIT::Matrix<fastEIT::dtype::real>* fastEIT::matrix::loadtxt<fastEIT::dtype::real>(const std::string, cudaStream_t);
template fastEIT::Matrix<fastEIT::dtype::index>* fastEIT::matrix::loadtxt<fastEIT::dtype::index>(const std::string, cudaStream_t);
template class fastEIT::Matrix<fastEIT::dtype::real>;
template class fastEIT::Matrix<fastEIT::dtype::index>;
