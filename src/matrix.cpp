// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "../include/fasteit.h"
#include "../include/matrix_kernel.h"

// create new matrix
template<
    class type
>
fastEIT::Matrix<type>::Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : host_data_(nullptr), device_data_(nullptr), rows_(rows), data_rows_(rows),
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
    if ((this->rows() % matrix::block_size != 0) && (this->rows() != 1)) {
        this->data_rows_ = (this->rows() / matrix::block_size + 1) * matrix::block_size;
    }
    if ((this->columns() % matrix::block_size != 0) && (this->columns() != 1)) {
        this->data_columns_ = (this->columns() / matrix::block_size + 1) * matrix::block_size;
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
template <
    class type
>
fastEIT::Matrix<type>::~Matrix() {
    // free matrix host data
    cudaFree(this->host_data_);

    // free matrix device data
    cudaFree(this->device_data_);
}

// copy matrix
template <
    class type
>
void fastEIT::Matrix<type>::copy(const std::shared_ptr<Matrix<type>> other,
    cudaStream_t stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("Matrix::copy: other = nullptr");
    }

    // check size
    if ((other->data_rows() != this->data_rows()) ||
        (other->data_columns() != this->data_columns())) {
        throw std::invalid_argument("Matrix::copy: size");
    }

    // copy data
    cudaMemcpyAsync(this->device_data(), other->device_data(),
        sizeof(type) * this->data_rows() * this->data_columns(),
        cudaMemcpyDeviceToDevice, stream);

    // TODO
    /*// check success
    if (error != cudaSuccess) {
        throw logic_error("Matrix::copyToDevice: copy error");
    }*/
}

// copy to device
template <
    class type
>
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
template <
    class type
>
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

// add matrix
template <
    class type
>
void fastEIT::Matrix<type>::add(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("Matrix::add: value = nullptr");
    }

    // check size
    if ((this->data_rows() != value->data_rows()) ||
        (this->data_columns() != value->data_columns())) {
        throw std::invalid_argument("Matrix::add: size");
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
void fastEIT::Matrix<type>::multiply(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cublasHandle_t handle,
    cudaStream_t stream) {
    throw std::logic_error("Matrix::multiply: not supported dtype");
}

// specialisation for dtype::real
namespace fastEIT {
    template <>
    void Matrix<dtype::real>::multiply(const std::shared_ptr<Matrix<dtype::real>> A,
        const std::shared_ptr<Matrix<dtype::real>> B, cublasHandle_t handle,
        cudaStream_t stream) {
        // check input
        if (A == nullptr) {
            throw std::invalid_argument("Matrix::multiply: A == nullptr");
        }
        if (B == nullptr) {
            throw std::invalid_argument("Matrix::multiply: B == nullptr");
        }
        if (handle == NULL) {
            throw std::invalid_argument("Matrix::multiply: handle == NULL");
        }

        // check size
        if ((A->data_columns() != B->data_rows()) ||
            (this->data_rows() != A->data_rows()) ||
            (this->data_columns() != B->data_columns())) {
            throw std::invalid_argument("Matrix::multiply: size");
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
                throw std::logic_error("Matrix::multiply: cublasSgemv");
            }
        }
        else {
            if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->data_rows(), B->data_columns(), A->data_columns(),
                &alpha, A->device_data(), A->data_rows(), B->device_data(), B->data_rows(), &beta,
                this->device_data(), this->data_rows()) != CUBLAS_STATUS_SUCCESS) {
                throw std::logic_error("Matrix::multiply: cublasSgemm");
            }
        }
    }
}

// scalar multiply matrix
template <
    class type
>
void fastEIT::Matrix<type>::scalarMultiply(type scalar, cudaStream_t stream) {
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
void fastEIT::Matrix<type>::vectorDotProduct(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cudaStream_t stream) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("Matrix::vectorDotProduct: A == nullptr");
    }
    if (B == nullptr) {
        throw std::invalid_argument("Matrix::vectorDotProduct: B == nullptr");
    }

    // check size
    if ((this->data_rows() != A->data_rows()) ||
        (this->data_rows() != B->data_rows())) {
        throw std::invalid_argument("Matrix::vectorDotProduct: size");
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
    this->sum(std::shared_ptr<fastEIT::Matrix<type>>(this, noop_deleter()), stream);
}

// sum
template <
    class type
>
void fastEIT::Matrix<type>::sum(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("Matrix::sum: value == nullptr");
    }

    // check size
    if (this->data_rows() != value->data_rows()) {
        throw std::invalid_argument("Matrix::sum: size");
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
void fastEIT::Matrix<type>::min(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("Matrix::min: value == nullptr");
    }

    // check size
    if (this->data_rows() != value->data_rows()) {
        throw std::invalid_argument("Matrix::min: size");
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
void fastEIT::Matrix<type>::max(const std::shared_ptr<Matrix<type>> value,
    cudaStream_t stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("Matrix::max: value == nullptr");
    }

    // check size
    if (this->data_rows() != value->data_rows()) {
        throw std::invalid_argument("Matrix::max: size");
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
std::shared_ptr<fastEIT::Matrix<type>> fastEIT::matrix::loadtxt(std::istream* istream,
    cudaStream_t stream) {
    // check input
    if (istream == nullptr) {
        throw std::invalid_argument("matrix::loadtxt: istream == nullptr");
    }

    // read lines
    std::vector<std::string> lines;
    std::string line;
    while (!istream->eof()) {
        // read line
        getline(*istream, line);

        // add line to vector
        lines.push_back(line);
    }

    // read values
    std::vector<std::vector<type> > values;
    for (auto line : lines) {
        // create row vector
        std::vector<type> row;
        type value;

        // create string stream
        std::stringstream line_stream(line);

        // read values of line
        while (!line_stream.eof()) {
            // check success
            if (line_stream.fail()) {
                throw std::logic_error("matrix::loadtxt: invalid value");
            }

            line_stream >> value;
            row.push_back(value);
        }

        // add row
        values.push_back(row);
    }

    // create matrix
    auto matrix = std::make_shared<Matrix<type>>(values.size() - 1, values[0].size(), stream);

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
std::shared_ptr<fastEIT::Matrix<type>> fastEIT::matrix::loadtxt(const std::string filename, cudaStream_t stream) {
    // open file stream
    std::ifstream file;
    file.open(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("matrix::loadtxt: cannot open file!");
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
void fastEIT::matrix::savetxt(const std::shared_ptr<Matrix<type>> matrix,
    std::ostream* ostream) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("matrix::savetxt: matrix == nullptr");
    }
    if (ostream == nullptr) {
        throw std::invalid_argument("matrix::savetxt: ostream == nullptr");
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
void fastEIT::matrix::savetxt(const std::string filename,
    const std::shared_ptr<Matrix<type>> matrix) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("matrix::savetxt: matrix == nullptr");
    }

    // open file stream
    std::ofstream file;
    file.open(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("matrix::savetxt: cannot open file!");
    }

    // save matrix
    savetxt<type>(matrix, &file);

    // close file
    file.close();
}

// specialisation
template std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
    fastEIT::matrix::loadtxt<fastEIT::dtype::real>(std::istream*, cudaStream_t);
template std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>
    fastEIT::matrix::loadtxt<fastEIT::dtype::index>(std::istream*, cudaStream_t);

template std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>
    fastEIT::matrix::loadtxt<fastEIT::dtype::real>(const std::string, cudaStream_t);
template std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>
    fastEIT::matrix::loadtxt<fastEIT::dtype::index>(const std::string, cudaStream_t);

template void fastEIT::matrix::savetxt<fastEIT::dtype::real>(
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>, std::ostream*);
template void fastEIT::matrix::savetxt<fastEIT::dtype::index>(
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>, std::ostream*);

template void fastEIT::matrix::savetxt<fastEIT::dtype::real>(const std::string,
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>);
template void fastEIT::matrix::savetxt<fastEIT::dtype::index>(const std::string,
    const std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::index>>);

template class fastEIT::Matrix<fastEIT::dtype::real>;
template class fastEIT::Matrix<fastEIT::dtype::index>;
