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

#include "json.h"
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
        throw std::runtime_error(
            str::format("mpFlow::numeric::Matrix::Matrix: cannot create device data memory: %f kB")(
                (double)(sizeof(type) * this->dataRows * this->dataCols) / 1024.0));
    }

    if (allocateHostMemory) {
        // create matrix host data memory
        error = cudaHostAlloc((void**)&this->hostData, sizeof(type) *
            this->dataRows * this->dataCols, cudaHostAllocDefault);

        CudaCheckError();
        if (error != cudaSuccess) {
            throw std::runtime_error(
                str::format("mpFlow::numeric::Matrix::Matrix: cannot create host data memory: %f kB")(
                    (double)(sizeof(type) * this->dataRows * this->dataCols) / 1024.0));
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
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::copy: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, other->rows, other->cols));
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

template <
    class type
>
void mpFlow::numeric::Matrix<type>::diag(std::shared_ptr<Matrix<type> const> const matrix,
    cudaStream_t const stream) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::diag: matrix == nullptr");
    }
    if ((this->rows != matrix->rows) || (this->cols != matrix->cols)) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::diag: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, matrix->rows, matrix->cols));
    }
    
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::diag(blocks, threads, stream, matrix->deviceData,
        this->dataRows, this->deviceData);
}

// add scalar to matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::add(type const value, cudaStream_t const stream) {
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);
 
    // call kernel
    matrixKernel::add(blocks, threads, stream, value,
        this->dataRows, this->deviceData);
}

// add matrix
template <
    class type
>
void mpFlow::numeric::Matrix<type>::add(std::shared_ptr<Matrix<type> const> const other,
    cudaStream_t const stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::add: value == nullptr");
    }

    // check size
    if ((this->rows != other->rows) || (this->cols != other->cols)) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::add: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, other->rows, other->cols));
    }

    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::add(blocks, threads, stream, other->deviceData,
        this->dataRows, this->deviceData);
}

// matrix multiply
template <
    class type
>
void mpFlow::numeric::Matrix<type>::multiply(std::shared_ptr<Matrix<type> const> const A,
    std::shared_ptr<Matrix<type> const> const B, cublasHandle_t const handle,
    cudaStream_t const stream, cublasOperation_t const transA, cublasOperation_t const transB) {
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

    // set cublas stream
    cublasSetStream(handle, stream);

    // multiply matrices
    type const alpha = 1.0, beta = 0.0;
    if (B->dataCols == 1) {
        if (cublasWrapper<type>::gemv(handle, transA, A->dataRows, A->dataCols, &alpha, A->deviceData,
            A->dataRows, B->deviceData, 1, &beta, this->deviceData, 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("mpFlow::numeric::Matrix::multiply: cublasSgemv");
        }
    }
    else {
        if (cublasWrapper<type>::gemm(handle, transA, transB, this->dataRows, this->dataCols,
            transA == CUBLAS_OP_T ? A->dataRows : A->dataCols,
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
    if (this->rows != A->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::elementwiseMultiply: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, A->rows, A->cols));
    }
    if (this->rows != B->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::elementwiseMultiply: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, B->rows, B->cols));
    }

    // get minimum colums
    unsigned columns = std::min(std::min(this->dataCols, A->dataCols), B->dataCols);

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
    if (this->rows != A->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::elementwiseDivision: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, A->rows, A->cols));
    }
    if (this->rows != B->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::elementwiseDivision: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, B->rows, B->cols));
    }

    // get minimum colums
    unsigned columns = std::min(std::min(this->dataCols, A->dataCols), B->dataCols);

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
    if (this->rows != A->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::vectorDotProduct: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, A->rows, A->cols));
    }
    if (this->rows != B->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::vectorDotProduct: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, B->rows, B->cols));
    }

    // get minimum colums
    unsigned const columns = std::min(std::min(this->dataCols, A->dataCols), B->dataCols);

    // kernel dimension
    dim3 const blocks(this->dataRows / matrix::blockSize,
        columns == 1 ? 1 : columns / matrix::blockSize);
    dim3 const threads(matrix::blockSize,
        columns == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::vectorDotProduct<type>(blocks, threads, stream,
        A->deviceData, B->deviceData, this->dataRows,
        this->deviceData);

    // sum
    struct noop_deleter { void operator()(void*) {} };
    this->sum(std::shared_ptr<mpFlow::numeric::Matrix<type>>(this, noop_deleter()), stream);
}

// set indexed elements of a matrix to a new value
template <
    class type
>
void mpFlow::numeric::Matrix<type>::setIndexedElements(std::shared_ptr<Matrix<unsigned> const> const indices,
    type const value, cudaStream_t const stream) {
    // check input
    if (indices == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::setIndexedElements: indices == nullptr");
    }
    
    // check shape
    if (indices->cols != this->cols) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::setIndexedElements: cols not equal: %d != %d")
            (this->cols, indices->cols));
    }
    
    // kernel dimensions
    dim3 const blocks(indices->dataRows / matrix::blockSize,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::blockSize);
    dim3 const threads(matrix::blockSize,
        this->dataCols == 1 ? 1 : matrix::blockSize);

    // call kernel
    matrixKernel::setIndexedElements<type>(blocks, threads, stream, indices->deviceData,
        indices->dataRows, value, this->dataRows, this->deviceData);
}

// sum
template <
    class type
>
void mpFlow::numeric::Matrix<type>::sum(std::shared_ptr<Matrix<type> const> const other,
    cudaStream_t const stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::sum: other == nullptr");
    }

    // check size
    if (this->rows != other->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::sum: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, other->rows, other->cols));
    }

    // get minimum columns
    unsigned columns = std::min(this->dataCols, other->dataCols);

    // perform prefix sum algorithm
    int size = this->dataRows;
    dim3 blocks((this->dataRows + 32 - 1) / 32,
        columns == 1 ? 1 : columns / matrix::blockSize);
    dim3 threads(32, columns == 1 ? 1 : matrix::blockSize);

    matrixKernel::sum<type>(blocks, threads, stream, other->deviceData,
        this->dataRows, size, this->deviceData);

    do {
        blocks.x = (blocks.x + 32 - 1) / 32;
        size = (size + 32 - 1) / 32;

        matrixKernel::sum(blocks, threads, stream, this->deviceData,
            this->dataRows, size, this->deviceData);
    }
    while (size >= 32);
}

// min
template <
    class type
>
void mpFlow::numeric::Matrix<type>::min(std::shared_ptr<Matrix<type> const> const other,
    cudaStream_t const stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::min: other == nullptr");
    }

    // check size
    if (this->rows != other->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::min: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, other->rows, other->cols));
    }

    // kernel settings
    unsigned blocks = this->dataRows / matrix::blockSize;
    unsigned offset = 1;

    // start kernel once
    matrixKernel::min<type>(blocks, matrix::blockSize, stream,
        other->deviceData, this->rows, offset, this->deviceData);

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
void mpFlow::numeric::Matrix<type>::max(std::shared_ptr<Matrix<type> const> const other,
    cudaStream_t const stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::max: other == nullptr");
    }

    // check size
    if (this->rows != other->rows) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::Matrix::max: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, other->rows, other->cols));
    }

    // kernel settings
    unsigned blocks = this->dataRows / matrix::blockSize;
    unsigned offset = 1;

    // start kernel once
    matrixKernel::max<type>(blocks, matrix::blockSize, stream,
        other->deviceData, this->rows, offset, this->deviceData);

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
        throw std::runtime_error(
            str::format("mpFlow::numeric::Matrix::savetxt: cannot open file: %s")(filename));
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

    // check for empty matrix
    if ((values.size() == 0) || (values[0].size() == 0)) {
        throw std::runtime_error("mpFlow::numeric::Matrix::loadtxt: cannot parse file!");
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
        throw std::runtime_error(
            str::format("mpFlow::numeric::Matrix::loadtxt: cannot open file: %s")(filename));
    }

    // load matrix from file
    try {
        auto const matrix = Matrix<type>::loadtxt(file, stream, delimiter);

        file.close();
        return matrix;
    }
    catch (std::exception const&) {
        file.close();
        throw std::runtime_error(
            str::format("mpFlow::numeric::Matrix::loadtxt: cannot parse file: %s")(filename));
    }
}

// converts matrix to eigen array
template <
    class type
>
Eigen::Array<typename mpFlow::typeTraits::convertComplexType<type>::type,
    Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::Matrix<type>::toEigen() const {
    if (this->hostData == nullptr) {
        throw std::runtime_error("mpFlow::numeric::Matrix::toEigen: host memory was not allocated");
    }

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

template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::Matrix<type>::fromJsonArray(
    json_value const& array, cudaStream_t const stream) {
    // check type of json value
    if (array.type != json_array) {
        return nullptr;
    }

    // exctract sizes
    unsigned const rows = array.u.array.length;
    unsigned const cols = array[0].type == json_array ? array[0].u.array.length : 1;

    // create matrix
    auto matrix = std::make_shared<mpFlow::numeric::Matrix<type>>(rows, cols, stream);

    // exctract values
    if (array[0].type != json_array) {
        for (unsigned row = 0; row < matrix->rows; ++row) {
            (*matrix)(row, 0) = array[row].u.dbl;
        }
    }
    else {
        for (unsigned row = 0; row < matrix->rows; ++row)
        for (unsigned col = 0; col < matrix->cols; ++col) {
            (*matrix)(row, col) = array[row][col].u.dbl;
        }
    }
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
