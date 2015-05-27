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

#include <string>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <fstream>

#include "mpflow/mpflow.h"
#include "mpflow/numeric/matrix_kernel.h"

// create new matrix
template <
    class type
>
mpFlow::numeric::Matrix<type>::Matrix(unsigned const rows, unsigned const cols,
    cudaStream_t const stream, type const value, bool const allocateHostMemory)
    : hostData(nullptr), deviceData(nullptr), rows(rows), cols(cols), dataRows(rows), dataCols(cols) {
    // check input
    if (rows == 0) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::Matrix: rows == 0");
    }
    if (cols == 0) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::Matrix: cols == 0");
    }

    // cuda error
    cudaError_t error = cudaSuccess;

    // correct size to block size
    if ((this->rows % matrix::blockSize != 0) && (this->rows != 1)) {
        this->dataRows = math::roundTo(this->rows, matrix::blockSize);
    }
    if ((this->cols % matrix::blockSize != 0) && (this->cols != 1)) {
        this->dataCols = math::roundTo(this->cols, matrix::blockSize);
    }

    // create matrix device data memory
    error = cudaMalloc((void**)&this->deviceData,
        sizeof(type) * this->dataRows * this->dataCols);

    CudaCheckError();
    if (error != cudaSuccess) {
        throw std::runtime_error("mpFlow::numeric::Matrix::Matrix: create device data memory");
    }

    if (allocateHostMemory) {
        // create matrix host data memory
        error = cudaHostAlloc((void**)&this->hostData, sizeof(type) *
            this->dataRows * this->dataCols, cudaHostAllocDefault);

        CudaCheckError();
        if (error != cudaSuccess) {
            throw std::runtime_error("mpFlow::numeric::Matrix::Matrix: create host data memory");
        }

        // init all data with zeros
        std::memset(this->hostData, 0, sizeof(type) * this->dataRows * this->dataCols);

        // when default value differs from zero, overwrite only used part of matrix
        if (value != type(0)) {
            for (unsigned row = 0; row < this->dataRows; ++row)
            for (unsigned col = 0; col < this->dataCols; ++col) {
                this->hostData[row + this->dataRows * col] = value;
            }
        }

        this->copyToDevice(stream);
        cudaStreamSynchronize(stream);
    }
    else {
        this->fill(value, stream);
    }
}

// release matrix
template <
    class type
>
mpFlow::numeric::Matrix<type>::~Matrix() {
    if (this->hostData != nullptr) {
        // free matrix host data
        CudaSafeCall(cudaFreeHost(this->hostData));
        CudaCheckError();
    }

    // free matrix device data
    CudaSafeCall(cudaFree(this->deviceData));
    CudaCheckError();
}

// helper function to create unit matrix
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>>
    mpFlow::numeric::Matrix<type>::eye(unsigned const size, cudaStream_t const stream) {
    auto matrix = std::make_shared<Matrix<type>>(size, size, stream);
    
    matrix->setEye(stream);
    matrix->copyToHost(stream);
    cudaStreamSynchronize(stream);

    return matrix;
}

// copy matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::copy(std::shared_ptr<Matrix<type> const> const other,
    cudaStream_t const stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::copy: other == nullptr");
    }

    // check size
    if ((other->rows != this->rows) || (other->cols != this->cols)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::copy: shape does not match");
    }

    // copy data
    CudaSafeCall(
        cudaMemcpyAsync(this->deviceData, other->deviceData,
            sizeof(type) * this->dataRows * this->dataCols,
            cudaMemcpyDeviceToDevice, stream));

    CudaCheckError();
}

// copy to device
template <
    class type
>
void mpFlow::numeric::Matrix<type>::copyToDevice(cudaStream_t const stream) {
    if (this->hostData == nullptr) {
        throw std::runtime_error("mpFlow::numeric::Matrix::copyToDevice: host memory was not allocated");
    }

    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->deviceData, this->hostData,
            sizeof(type) * this->dataRows * this->dataCols,
            cudaMemcpyHostToDevice, stream));

    CudaCheckError();
}

// copy to host
template <
    class type
>
void mpFlow::numeric::Matrix<type>::copyToHost(cudaStream_t const stream) const {
    if (this->hostData == nullptr) {
        throw std::runtime_error("mpFlow::numeric::Matrix::copyToHost: host memory was not allocated");
    }

    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->hostData, this->deviceData,
            sizeof(type) * this->dataRows * this->dataCols,
            cudaMemcpyDeviceToHost, stream));

    CudaCheckError();
}

template <
    class type
>
void mpFlow::numeric::Matrix<type>::fill(type const value, cudaStream_t const stream) {
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::fill(blocks, threads, stream, value,
        this->rows, this->cols, this->dataRows, this->deviceData);
}

template <
    class type
>
void mpFlow::numeric::Matrix<type>::setEye(cudaStream_t const stream) {
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::setEye(blocks, threads, stream, this->rows,
        this->dataRows, this->deviceData);
}

// add matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::add(std::shared_ptr<Matrix<type> const> const value,
    cudaStream_t const stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::add: value == nullptr");
    }

    // check size
    if ((this->rows != value->rows) ||
        (this->cols != value->cols)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::add: shape does not match");
    }

    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::add(blocks, threads, stream, value->deviceData,
        this->dataRows, this->deviceData);
}


// matrix multiply
template <
    class type
>
void mpFlow::numeric::Matrix<type>::multiply(std::shared_ptr<Matrix<type> const> const A,
    std::shared_ptr<Matrix<type> const> const B, cublasHandle_t const handle,
    cudaStream_t const stream) {
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
    if ((A->cols != B->rows) ||
        (this->rows != A->rows) ||
        (this->cols != B->cols)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::multiply: shape does not match");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // multiply matrices
    type alpha = 1.0f;
    type beta = 0.0f;

    if (B->dataCols == 1) {
        if (cublasWrapper<type>::gemv(handle, CUBLAS_OP_N, A->dataRows, A->dataCols, &alpha, A->deviceData,
            A->dataRows, B->deviceData, 1, &beta, this->deviceData, 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("mpFlow::numeric::Matrix::multiply: cublasSgemv");
        }
    }
    else {
        if (cublasWrapper<type>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->dataRows, B->dataCols, A->dataCols,
            &alpha, A->deviceData, A->dataRows, B->deviceData, B->dataRows, &beta,
            this->deviceData, this->dataRows) != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("mpFlow::numeric::Matrix::multiply: cublasSgemm");
        }
    }
}

// specialisation for sparse matrices
template <
    class type
>
void mpFlow::numeric::Matrix<type>::multiply(std::shared_ptr<SparseMatrix<type> const> const A,
    std::shared_ptr<Matrix<type> const> const B, cublasHandle_t const,
    cudaStream_t const stream) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::multiply: A == nullptr");
    }

    struct noop_deleter { void operator()(void*) {} };
    A->multiply(B, stream, std::shared_ptr<mpFlow::numeric::Matrix<type>>(this, noop_deleter()));
}

// scalar multiply matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::scalarMultiply(type const scalar,
    cudaStream_t const stream) {
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::scale<type>(blocks, threads, stream, scalar,
        this->dataRows, this->deviceData);
}

// elementwise multiply
template <
    class type
>
void mpFlow::numeric::Matrix<type>::elementwiseMultiply(
    std::shared_ptr<Matrix<type> const> const A,
    std::shared_ptr<Matrix<type> const> const B, cudaStream_t const stream) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::elementwiseMultiply: A == nullptr");
    }
    if (B == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::elementwiseMultiply: B == nullptr");
    }

    // check size
    if ((this->rows != A->rows) ||
        (this->rows != B->rows)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::elementwiseMultiply: shape does not match");
    }

    // get minimum colums
    unsigned columns = std::min(std::min(this->dataCols,
        A->dataCols), B->dataCols);

    // kernel dimension
    dim3 blocks(this->dataRows / matrix::blockSize,
        columns == 1 ? 1 : columns / matrix::blockSize);
    dim3 threads(matrix::blockSize,
        columns == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::elementwiseMultiply<type>(blocks, threads, stream,
        A->deviceData, B->deviceData, this->dataRows,
        this->deviceData);
}

// elementwise division
template <
    class type
>
void mpFlow::numeric::Matrix<type>::elementwiseDivision(
    std::shared_ptr<Matrix<type> const> const A,
    std::shared_ptr<Matrix<type> const> const B, cudaStream_t const stream) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::elementwiseDivision: A == nullptr");
    }
    if (B == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::elementwiseDivision: B == nullptr");
    }

    // check size
    if ((this->rows != A->rows) ||
        (this->rows != B->rows)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::elementwiseDivision: shape does not match");
    }

    // get minimum colums
    unsigned columns = std::min(std::min(this->dataCols,
        A->dataCols), B->dataCols);

    // kernel dimension
    dim3 blocks(this->dataRows / matrix::blockSize,
        columns == 1 ? 1 : columns / matrix::blockSize);
    dim3 threads(matrix::blockSize,
        columns == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::elementwiseDivision<type>(blocks, threads, stream,
        A->deviceData, B->deviceData, this->dataRows,
        this->deviceData);
}

// vector dot product
template <
    class type
>
void mpFlow::numeric::Matrix<type>::vectorDotProduct(
    std::shared_ptr<Matrix<type> const> const A,
    std::shared_ptr<Matrix<type> const> const B, cudaStream_t const stream) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::vectorDotProduct: A == nullptr");
    }
    if (B == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::vectorDotProduct: B == nullptr");
    }

    // check size
    if ((this->rows != A->rows) ||
        (this->rows != B->rows)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::vectorDotProduct: shape does not match");
    }

    // get minimum colums
    unsigned columns = std::min(std::min(this->dataCols,
        A->dataCols), B->dataCols);

    // kernel dimension
    dim3 blocks(this->dataRows / matrix::blockSize,
        columns == 1 ? 1 : columns / matrix::blockSize);
    dim3 threads(matrix::blockSize,
        columns == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::vectorDotProduct<type>(blocks, threads, stream,
        A->deviceData, B->deviceData, this->dataRows,
        this->deviceData);

    // sum
    struct noop_deleter { void operator()(void*) {} };
    this->sum(std::shared_ptr<mpFlow::numeric::Matrix<type>>(this, noop_deleter()), stream);
}

// sum
template <
    class type
>
void mpFlow::numeric::Matrix<type>::sum(std::shared_ptr<Matrix<type> const> const value,
    cudaStream_t const stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::sum: value == nullptr");
    }

    // check size
    if (this->rows != value->rows) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::sum: shape does not match");
    }

    // get minimum columns
    unsigned columns = std::min(this->dataCols, value->dataCols);

    // kernel settings
    dim3 blocks(this->dataRows / matrix::blockSize,
        columns == 1 ? 1 : columns / matrix::blockSize);
    dim3 threads(matrix::blockSize,
        columns == 1 ? 1 : matrix::blockSize);
    unsigned offset = 1;

    // start kernel once
    matrixKernel::sum<type>(blocks, threads, stream, value->deviceData,
        this->dataRows, offset, this->deviceData);

    // start kernel
    do {
        // update settings
        offset *= matrix::blockSize;
        blocks.x = (blocks.x + matrix::blockSize - 1) /
            matrix::blockSize;

        matrixKernel::sum(blocks, threads, stream, this->deviceData,
            this->dataRows, offset, this->deviceData);

    }
    while (offset * matrix::blockSize < this->dataRows);
}

// min
template <
    class type
>
void mpFlow::numeric::Matrix<type>::min(std::shared_ptr<Matrix<type> const> const value,
    cudaStream_t const stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::min: value == nullptr");
    }

    // check size
    if (this->rows != value->rows) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::min: shape does not match");
    }

    // kernel settings
    unsigned blocks = this->dataRows / matrix::blockSize;
    unsigned offset = 1;

    // start kernel once
    matrixKernel::min<type>(blocks, matrix::blockSize, stream,
        value->deviceData, this->rows, offset, this->deviceData);

    // start kernel
    do {
        // update settings
        offset *= matrix::blockSize;
        blocks = (blocks + matrix::blockSize - 1) / matrix::blockSize;

        matrixKernel::min<type>(blocks, matrix::blockSize, stream,
            this->deviceData, this->rows, offset, this->deviceData);

    }
    while (offset * matrix::blockSize < this->dataRows);
}

namespace mpFlow {
namespace numeric {
    template <>
    void Matrix<thrust::complex<float>>::min(
        std::shared_ptr<Matrix<thrust::complex<float>> const> const,
        cudaStream_t const) {
        throw std::runtime_error("mpFlow::numeric::Matrix::min: not possible for complex values");
    }

    template <>
    void Matrix<thrust::complex<double>>::min(
        std::shared_ptr<Matrix<thrust::complex<double>> const> const,
        cudaStream_t const) {
        throw std::runtime_error("mpFlow::numeric::Matrix::min: not possible for complex values");
    }
}
}

// max
template <
    class type
>
void mpFlow::numeric::Matrix<type>::max(std::shared_ptr<Matrix<type> const> const value,
    cudaStream_t const stream) {
    // check input
    if (value == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::max: value == nullptr");
    }

    // check size
    if (this->rows != value->rows) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::max: shape does not match");
    }

    // kernel settings
    unsigned blocks = this->dataRows / matrix::blockSize;
    unsigned offset = 1;

    // start kernel once
    matrixKernel::max<type>(blocks, matrix::blockSize, stream,
        value->deviceData, this->rows, offset, this->deviceData);

    // start kernel
    do {
        // update settings
        offset *= matrix::blockSize;
        blocks = (blocks + matrix::blockSize - 1) / matrix::blockSize;

        matrixKernel::max<type>(blocks, matrix::blockSize, stream,
            this->deviceData, this->rows, offset, this->deviceData);

    }
    while (offset * matrix::blockSize < this->dataRows);
}

namespace mpFlow {
namespace numeric {
    template <>
    void Matrix<thrust::complex<float>>::max(
        std::shared_ptr<Matrix<thrust::complex<float>> const> const,
        cudaStream_t const) {
        throw std::runtime_error("mpFlow::numeric::Matrix::max: not possible for complex values");
    }

    template <>
    void Matrix<thrust::complex<double>>::max(
        std::shared_ptr<Matrix<thrust::complex<double>> const> const,
        cudaStream_t const) {
        throw std::runtime_error("mpFlow::numeric::Matrix::max: not possible for complex values");
    }
}
}

// save matrix to stream
template <
    class type
>
void mpFlow::numeric::Matrix<type>::savetxt(std::ostream& ostream, char const delimiter) const {
    // check input
    if (this->hostData == nullptr) {
        throw std::runtime_error("mpFlow::numeric::Matrix::savetxt: host memory was not allocated");
    }

    // write data
    for (unsigned row = 0; row < this->rows; ++row) {
        for (unsigned column = 0; column < this->cols - 1; ++column) {
            ostream << (*this)(row, column) << delimiter;
        }
        ostream << (*this)(row, this->cols - 1);

        // print new line not on last line
        if (row != this->rows - 1) {
            ostream << std::endl;
        }
    }
}

// save matrix to file
template <
    class type
>
void mpFlow::numeric::Matrix<type>::savetxt(std::string const filename, char const delimiter) const {
    // check input
    if (this->hostData == nullptr) {
        throw std::runtime_error("mpFlow::numeric::Matrix::savetxt: host memory was not allocated");
    }

    // open file stream
    std::ofstream file(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::runtime_error("mpFlow::numeric::Matrix::savetxt: cannot open file!");
    }

    // save matrix
    file.precision(std::numeric_limits<double>::digits10 + 1);
    this->savetxt(file, delimiter);

    // close file
    file.close();
}

// load matrix from stream
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::Matrix<type>::loadtxt(
    std::istream& istream, cudaStream_t const stream, char const delimiter) {
    // read matrix
    std::vector<std::vector<type>> values;
    std::string line;
    type value;
    while (!istream.eof()) {
        // read line
        getline(istream, line);

        // check succes
        if (istream.fail()) {
            break;
        }

        // create string stream
        std::stringstream line_stream(line);

        // read values of line
        std::vector<type> row;
        while (!line_stream.eof()) {
            // read value
            std::string token;
            std::getline(line_stream, token, delimiter);

            // extract value
            std::stringstream valueStream(token);
            valueStream >> value;

            // check read error
            if (line_stream.bad()) {
                throw std::runtime_error("mpFlow::numeric::Matrix::loadtxt: invalid value");
            }
            else if (!valueStream.fail()) {
                row.push_back(value);
            }
        }

        // add row
        if (row.size() != 0) {
            values.push_back(row);
        }
    }

    // covert STL vector to Matrix
    auto matrix = std::make_shared<Matrix<type>>(values.size(), values[0].size(), stream);
    for (unsigned row = 0; row < matrix->rows; ++row)
    for (unsigned col = 0; col < matrix->cols; ++col) {
        (*matrix)(row, col) = values[row][col];
    }
    matrix->copyToDevice(stream);

    return matrix;
}

// load matrix from file
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::Matrix<type>::loadtxt(
    std::string const filename, cudaStream_t const stream, char const delimiter) {
    // open file stream
    std::ifstream file(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::runtime_error("mpFlow::numeric::Matrix::loadtxt: cannot open file!");
    }

    // load matrix
    auto matrix = Matrix<type>::loadtxt(file, stream, delimiter);

    // close file
    file.close();

    return matrix;
}

// converts matrix to eigen array
template <
    class type
>
Eigen::Array<typename mpFlow::typeTraits::convertComplexType<type>::type,
    Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::Matrix<type>::toEigen() const {
    // create eigen array with mpflow_type
    Eigen::Array<typename typeTraits::convertComplexType<type>::type,
        Eigen::Dynamic, Eigen::Dynamic> array(this->dataRows, this->dataCols);

    // copy data
    memcpy(array.data(), this->hostData, sizeof(type) *
        array.rows() * array.cols());

    // resize array
    array.conservativeResize(this->rows, this->cols);

    return array;
}

// converts eigen array to matrix
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::Matrix<type>::fromEigen(
    Eigen::Ref<Eigen::Array<typename typeTraits::convertComplexType<type>::type,
    Eigen::Dynamic, Eigen::Dynamic> const> const array, cudaStream_t const stream) {
    // copy array into changable intermediate array
    Eigen::Array<typename typeTraits::convertComplexType<type>::type,
        Eigen::Dynamic, Eigen::Dynamic> tempArray = array;

    // create mpflow matrix and resize eigen array to correct size
    auto matrix = std::make_shared<Matrix<type>>(array.rows(),
        array.cols(), stream);
    tempArray.conservativeResize(matrix->dataRows, matrix->dataCols);

    // copy data
    memcpy(matrix->hostData, tempArray.data(), sizeof(type) *
        matrix->dataRows * matrix->dataCols);

    // copy data to device
    matrix->copyToDevice(stream);

    return matrix;
}

// specialisation
template class mpFlow::numeric::Matrix<float>;
template class mpFlow::numeric::Matrix<double>;
template class mpFlow::numeric::Matrix<thrust::complex<float>>;
template class mpFlow::numeric::Matrix<thrust::complex<double>>;
template class mpFlow::numeric::Matrix<unsigned>;
template class mpFlow::numeric::Matrix<int>;
