// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "mpflow/mpflow.h"
#include "mpflow/numeric/matrix_kernel.h"

// create new matrix
template<
    class type
>
mpFlow::numeric::Matrix<type>::Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream,
    type value)
    : host_data_(nullptr), device_data_(nullptr), rows_(rows), columns_(columns),
        data_rows_(rows), data_columns_(columns) {
    // check input
    if (rows == 0) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::Matrix: rows == 0");
    }
    if (columns == 0) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::Matrix: columns == 0");
    }

    // cuda error
    cudaError_t error = cudaSuccess;

    // correct size to block size
    if ((this->rows() % matrix::block_size != 0) && (this->rows() != 1)) {
        this->data_rows_ = math::roundTo(this->rows(), matrix::block_size);
    }
    if ((this->columns() % matrix::block_size != 0) && (this->columns() != 1)) {
        this->data_columns_ = math::roundTo(this->columns(), matrix::block_size);
    }

    // create matrix host data memory
    error = cudaHostAlloc((void**)&this->host_data_, sizeof(type) *
        this->data_rows() * this->data_columns(), cudaHostAllocDefault);

    CudaCheckError();

    // check success
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::Matrix::Matrix: create host data memory");
    }

    // create matrix device data memory
    error = cudaMalloc((void**)&this->device_data_,
        sizeof(type) * this->data_rows() * this->data_columns());

    CudaCheckError();

    // check success
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::Matrix::Matrix: create device data memory");
    }

    // init data with 0.0
    for (dtype::size i = 0; i < this->data_rows(); i++) {
        for (dtype::size j = 0; j < this->data_columns(); j++) {
            this->host_data()[i + this->data_rows() * j] = value;
        }
    }
    this->copyToDevice(stream);
}

// release matrix
template <
    class type
>
mpFlow::numeric::Matrix<type>::~Matrix() {
    // free matrix host data
    cudaFreeHost(this->host_data_);
    CudaCheckError();

    // free matrix device data
    cudaFree(this->device_data_);
    CudaCheckError();
}

// copy matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::copy(const std::shared_ptr<Matrix<type>> other,
    cudaStream_t stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::copy: other == nullptr");
    }

    // check size
    if ((other->rows() != this->rows()) ||
        (other->columns() != this->columns())) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::copy: shape does not match");
    }

    // copy data
    CudaSafeCall(
        cudaMemcpyAsync(this->device_data(), other->device_data(),
        sizeof(type) * this->data_rows() * this->data_columns(),
        cudaMemcpyDeviceToDevice, stream));

    CudaCheckError();
}

// copy to device
template <
    class type
>
void mpFlow::numeric::Matrix<type>::copyToDevice(cudaStream_t stream) {
    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->device_data(), this->host_data(),
            sizeof(type) * this->data_rows() * this->data_columns(),
            cudaMemcpyHostToDevice, stream));

    CudaCheckError();
}

// copy to host
template <
    class type
>
void mpFlow::numeric::Matrix<type>::copyToHost(cudaStream_t stream) {
    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->host_data(), this->device_data(),
            sizeof(type) * this->data_rows() * this->data_columns(),
            cudaMemcpyDeviceToHost, stream));

    CudaCheckError();
}

// add matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::add(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::add: value == nullptr");
    }

    // check size
    if ((this->rows() != value->rows()) ||
        (this->columns() != value->columns())) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::add: shape does not match");
    }

    // dimension
    dim3 blocks(this->data_rows() == 1 ? 1 : this->data_rows() / matrix::block_size,
        this->data_columns() == 1 ? 1 : this->data_columns() / matrix::block_size);
    dim3 threads(this->data_rows() == 1 ? 1 : matrix::block_size,
        this->data_columns() == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::add(blocks, threads, stream, value->device_data(),
        this->data_rows(), this->device_data());
}


// matrix multiply
template <
    class type
>
void mpFlow::numeric::Matrix<type>::multiply(const std::shared_ptr<Matrix<type>>,
    const std::shared_ptr<Matrix<type>>, cublasHandle_t, cudaStream_t) {
    throw std::logic_error("mpFlow::numeric::Matrix::multiply: not supported dtype");
}

// specialisation for dtype::real
namespace mpFlow {
    template <>
    void numeric::Matrix<dtype::real>::multiply(const std::shared_ptr<Matrix<dtype::real>> A,
        const std::shared_ptr<Matrix<dtype::real>> B, cublasHandle_t handle,
        cudaStream_t stream) {
        // check input
        if (A == nullptr) {
            throw std::invalid_argument("mpFlow::numeric::Matrix::multiply: A == nullptr");
        }
        if (B == nullptr) {
            throw std::invalid_argument("mpFlow::numeric::Matrix::multiply: B == nullptr");
        }
        if (handle == NULL) {
            throw std::invalid_argument("mpFlow::numeric::Matrix::multiply: handle == NULL");
        }

        // check size
        if ((A->columns() != B->rows()) ||
            (this->rows() != A->rows()) ||
            (this->columns() != B->columns())) {
            throw std::invalid_argument("mpFlow::numeric::Matrix::multiply: shape does not match");
        }

        // set cublas stream
        cublasSetStream(handle, stream);

        // multiply matrices
        dtype::real alpha = 1.0f;
        dtype::real beta = 0.0f;

        if (B->data_columns() == 1) {
            if (cublasSgemv(handle, CUBLAS_OP_N, A->data_rows(), A->data_columns(), &alpha, A->device_data(),
                A->data_rows(), B->device_data(), 1, &beta, this->device_data(), 1)
                != CUBLAS_STATUS_SUCCESS) {
                throw std::logic_error("mpFlow::numeric::Matrix::multiply: cublasSgemv");
            }
        }
        else {
            if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->data_rows(), B->data_columns(), A->data_columns(),
                &alpha, A->device_data(), A->data_rows(), B->device_data(), B->data_rows(), &beta,
                this->device_data(), this->data_rows()) != CUBLAS_STATUS_SUCCESS) {
                throw std::logic_error("mpFlow::numeric::Matrix::multiply: cublasSgemm");
            }
        }
    }
}

// scalar multiply matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::scalarMultiply(type scalar, cudaStream_t stream) {
    // dimension
    dim3 blocks(this->data_rows() == 1 ? 1 : this->data_rows() / matrix::block_size,
        this->data_columns() == 1 ? 1 : this->data_columns() / matrix::block_size);
    dim3 threads(this->data_rows() == 1 ? 1 : matrix::block_size,
        this->data_columns() == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::scale<type>(blocks, threads, stream, scalar,
        this->data_rows(), this->device_data());
}

// vector dot product
template <
    class type
>
void mpFlow::numeric::Matrix<type>::vectorDotProduct(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cudaStream_t stream) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::vectorDotProduct: A == nullptr");
    }
    if (B == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::vectorDotProduct: B == nullptr");
    }

    // check size
    if ((this->rows() != A->rows()) ||
        (this->rows() != B->rows())) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::vectorDotProduct: shape does not match");
    }

    // get minimum colums
    dtype::size columns = std::min(std::min(this->data_columns(),
        A->data_columns()), B->data_columns());

    // kernel dimension
    dim3 blocks(this->data_rows() / matrix::block_size,
        columns == 1 ? 1 : columns / matrix::block_size);
    dim3 threads(matrix::block_size,
        columns == 1 ? 1 : matrix::block_size);

    // call dot kernel
    matrixKernel::vectorDotProduct<type>(blocks, threads, stream,
        A->device_data(), B->device_data(), this->data_rows(),
        this->device_data());

    // sum
    struct noop_deleter { void operator()(void*) {} };
    this->sum(std::shared_ptr<mpFlow::numeric::Matrix<type>>(this, noop_deleter()), stream);
}

// sum
template <
    class type
>
void mpFlow::numeric::Matrix<type>::sum(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::sum: value == nullptr");
    }

    // check size
    if (this->rows() != value->rows()) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::sum: shape does not match");
    }

    // get minimum columns
    dtype::size columns = std::min(this->data_columns(), value->data_columns());

    // kernel settings
    dim3 blocks(this->data_rows() / matrix::block_size,
        columns == 1 ? 1 : columns / matrix::block_size);
    dim3 threads(matrix::block_size,
        columns == 1 ? 1 : matrix::block_size);
    dtype::size offset = 1;

    // start kernel once
    matrixKernel::sum<type>(blocks, threads, stream, value->device_data(),
        this->data_rows(), offset, this->device_data());

    // start kernel
    do {
        // update settings
        offset *= matrix::block_size;
        blocks.x = (blocks.x + matrix::block_size - 1) /
            matrix::block_size;

        matrixKernel::sum(blocks, threads, stream, this->device_data(),
            this->data_rows(), offset, this->device_data());

    }
    while (offset * matrix::block_size < this->data_rows());
}

// min
template <
    class type
>
void mpFlow::numeric::Matrix<type>::min(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::min: value == nullptr");
    }

    // check size
    if (this->rows() != value->rows()) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::min: shape does not match");
    }

    // kernel settings
    dtype::size blocks = this->data_rows() / matrix::block_size;
    dtype::size offset = 1;

    // start kernel once
    matrixKernel::min<type>(blocks, matrix::block_size, stream,
        value->device_data(), this->rows(), offset, this->device_data());

    // start kernel
    do {
        // update settings
        offset *= matrix::block_size;
        blocks = (blocks + matrix::block_size - 1) / matrix::block_size;

        matrixKernel::min<type>(blocks, matrix::block_size, stream,
            this->device_data(), this->rows(), offset, this->device_data());

    }
    while (offset * matrix::block_size < this->data_rows());
}

// max
template <
    class type
>
void mpFlow::numeric::Matrix<type>::max(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::max: value == nullptr");
    }

    // check size
    if (this->rows() != value->rows()) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::max: shape does not match");
    }

    // kernel settings
    dtype::size blocks = this->data_rows() / matrix::block_size;
    dtype::size offset = 1;

    // start kernel once
    matrixKernel::max<type>(blocks, matrix::block_size, stream,
        value->device_data(), this->rows(), offset, this->device_data());

    // start kernel
    do {
        // update settings
        offset *= matrix::block_size;
        blocks = (blocks + matrix::block_size - 1) / matrix::block_size;

        matrixKernel::max<type>(blocks, matrix::block_size, stream,
            this->device_data(), this->rows(), offset, this->device_data());

    }
    while (offset * matrix::block_size < this->data_rows());
}

// load matrix from stream
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::matrix::loadtxt(std::istream* istream,
    cudaStream_t stream) {
    // check input
    if (istream == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::matrix::loadtxt: istream == nullptr");
    }

    // read matrix
    std::vector<std::vector<type>> values;
    std::string line;
    type value;
    while (!istream->eof()) {
        // read line
        getline(*istream, line);

        // check succes
        if (istream->fail()) {
            break;
        }

        // create string stream
        std::stringstream line_stream(line);

        // read values of line
        std::vector<type> row;
        while (!line_stream.eof()) {
            // read value
            line_stream >> value;

            // check read error
            if (line_stream.bad()) {
                throw std::logic_error("mpFlow::numeric::matrix::loadtxt: invalid value");
            }

            row.push_back(value);
        }

        // add row
        if (row.size() != 0) {
            values.push_back(row);
        }
    }

    // create matrix
    auto matrix = std::make_shared<Matrix<type>>(values.size(), values[0].size(), stream);

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

// load matrix from file
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::matrix::loadtxt(
    const std::string filename, cudaStream_t stream) {
    // open file stream
    std::ifstream file;
    file.open(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("mpFlow::numeric::matrix::loadtxt: cannot open file!");
    }

    // load matrix
    auto matrix = loadtxt<type>(&file, stream);

    // close file
    file.close();

    return matrix;
}

// save matrix to stream
template <
    class type
>
void mpFlow::numeric::matrix::savetxt(const std::shared_ptr<Matrix<type>> matrix,
    std::ostream* ostream) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::matrix::savetxt: matrix == nullptr");
    }
    if (ostream == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::matrix::savetxt: ostream == nullptr");
    }

    // write data
    for (dtype::index row = 0; row < matrix->rows(); ++row) {
        for (dtype::index column = 0; column < matrix->columns() - 1; ++column) {
            *ostream << (*matrix)(row, column) << " ";
        }
        *ostream << (*matrix)(row, matrix->columns() - 1) << std::endl;
    }
}

// save matrix to file
template <
    class type
>
void mpFlow::numeric::matrix::savetxt(const std::string filename,
    const std::shared_ptr<Matrix<type>> matrix) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::matrix::savetxt: matrix == nullptr");
    }

    // open file stream
    std::ofstream file;
    file.open(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("mpFlow::numeric::matrix::savetxt: cannot open file!");
    }

    // save matrix
    savetxt<type>(matrix, &file);

    // close file
    file.close();
}

// specialisation
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::numeric::matrix::loadtxt<mpFlow::dtype::real>(std::istream*, cudaStream_t);
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>
    mpFlow::numeric::matrix::loadtxt<mpFlow::dtype::index>(std::istream*, cudaStream_t);

template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::numeric::matrix::loadtxt<mpFlow::dtype::real>(const std::string, cudaStream_t);
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>
    mpFlow::numeric::matrix::loadtxt<mpFlow::dtype::index>(const std::string, cudaStream_t);

template void mpFlow::numeric::matrix::savetxt<mpFlow::dtype::real>(
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>, std::ostream*);
template void mpFlow::numeric::matrix::savetxt<mpFlow::dtype::index>(
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>, std::ostream*);

template void mpFlow::numeric::matrix::savetxt<mpFlow::dtype::real>(const std::string,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::numeric::matrix::savetxt<mpFlow::dtype::index>(const std::string,
    const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>);

template class mpFlow::numeric::Matrix<mpFlow::dtype::real>;
template class mpFlow::numeric::Matrix<mpFlow::dtype::index>;
