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

#include <string>
#include <sstream>
#include <iostream>
#include <fstream>

#include "mpflow/mpflow.h"
#include "mpflow/numeric/matrix_kernel.h"

// zero copy is not used by default
// bool mpFlow::numeric::Matrix<type>::useZeroCopy = false;

// create new matrix
template <
    class type
>
mpFlow::numeric::Matrix<type>::Matrix(dtype::size rows, dtype::size cols,
    cudaStream_t stream, type value, bool allocateHostMemory)
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
    if ((this->rows % matrix::block_size != 0) && (this->rows != 1)) {
        this->dataRows = math::roundTo(this->rows, matrix::block_size);
    }
    if ((this->cols % matrix::block_size != 0) && (this->cols != 1)) {
        this->dataCols = math::roundTo(this->cols, matrix::block_size);
    }

    // create matrix device data memory
    error = cudaMalloc((void**)&this->deviceData,
        sizeof(type) * this->dataRows * this->dataCols);

    CudaCheckError();
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::Matrix::Matrix: create device data memory");
    }

    if (allocateHostMemory) {
        // create matrix host data memory
        error = cudaHostAlloc((void**)&this->hostData, sizeof(type) *
            this->dataRows * this->dataCols, cudaHostAllocDefault);

        CudaCheckError();
        if (error != cudaSuccess) {
            throw std::logic_error("mpFlow::numeric::Matrix::Matrix: create host data memory");
        }

        // init data with default value
        for (dtype::size row = 0; row < this->dataRows; ++row)
        for (dtype::size col = 0; col < this->dataCols; ++col) {
            this->hostData[row + this->dataRows * col] = value;
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
        cudaFreeHost(this->hostData);
        CudaCheckError();
    }

    // free matrix device data
    cudaFree(this->deviceData);
    CudaCheckError();
}

template <
    class type
>
void mpFlow::numeric::Matrix<type>::fill(type value, cudaStream_t stream) {
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::block_size,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::block_size);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::block_size,
        this->dataCols == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::fill(blocks, threads, stream, value,
        this->dataRows, this->deviceData);
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
    if ((other->rows != this->rows) ||
        (other->cols != this->cols)) {
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
void mpFlow::numeric::Matrix<type>::copyToDevice(cudaStream_t stream) {
    if (this->hostData == nullptr) {
        throw std::logic_error("mpFlow::numeric::Matrix::copyToDevice: host memory was not allocated");
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
void mpFlow::numeric::Matrix<type>::copyToHost(cudaStream_t stream) {
    if (this->hostData == nullptr) {
        throw std::logic_error("mpFlow::numeric::Matrix::copyToHost: host memory was not allocated");
    }

    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->hostData, this->deviceData,
            sizeof(type) * this->dataRows * this->dataCols,
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
    if ((this->rows != value->rows) ||
        (this->cols != value->cols)) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::add: shape does not match");
    }

    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::block_size,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::block_size);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::block_size,
        this->dataCols == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::add(blocks, threads, stream, value->deviceData,
        this->dataRows, this->deviceData);
}


// matrix multiply
template <
    class type
>
void mpFlow::numeric::Matrix<type>::multiply(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cublasHandle_t handle, cudaStream_t stream) {
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
            throw std::logic_error("mpFlow::numeric::Matrix::multiply: cublasSgemv");
        }
    }
    else {
        if (cublasWrapper<type>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, A->dataRows, B->dataCols, A->dataCols,
            &alpha, A->deviceData, A->dataRows, B->deviceData, B->dataRows, &beta,
            this->deviceData, this->dataRows) != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error("mpFlow::numeric::Matrix::multiply: cublasSgemm");
        }
    }
}

// specialisation for sparse matrices
template <
    class type
>
void mpFlow::numeric::Matrix<type>::multiply(const std::shared_ptr<SparseMatrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cublasHandle_t, cudaStream_t stream) {
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
void mpFlow::numeric::Matrix<type>::scalarMultiply(type scalar, cudaStream_t stream) {
    // dimension
    dim3 blocks(this->dataRows == 1 ? 1 : this->dataRows / matrix::block_size,
        this->dataCols == 1 ? 1 : this->dataCols / matrix::block_size);
    dim3 threads(this->dataRows == 1 ? 1 : matrix::block_size,
        this->dataCols == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::scale<type>(blocks, threads, stream, scalar,
        this->dataRows, this->deviceData);
}

// elementwise multiply
template <
    class type
>
void mpFlow::numeric::Matrix<type>::elementwiseMultiply(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cudaStream_t stream) {
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
    dtype::size columns = std::min(std::min(this->dataCols,
        A->dataCols), B->dataCols);

    // kernel dimension
    dim3 blocks(this->dataRows / matrix::block_size,
        columns == 1 ? 1 : columns / matrix::block_size);
    dim3 threads(matrix::block_size,
        columns == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::elementwiseMultiply<type>(blocks, threads, stream,
        A->deviceData, B->deviceData, this->dataRows,
        this->deviceData);
}

// elementwise division
template <
    class type
>
void mpFlow::numeric::Matrix<type>::elementwiseDivision(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cudaStream_t stream) {
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
    dtype::size columns = std::min(std::min(this->dataCols,
        A->dataCols), B->dataCols);

    // kernel dimension
    dim3 blocks(this->dataRows / matrix::block_size,
        columns == 1 ? 1 : columns / matrix::block_size);
    dim3 threads(matrix::block_size,
        columns == 1 ? 1 : matrix::block_size);

    // call kernel
    matrixKernel::elementwiseDivision<type>(blocks, threads, stream,
        A->deviceData, B->deviceData, this->dataRows,
        this->deviceData);
}

// vector dot product
template <
    class type
>
void mpFlow::numeric::Matrix<type>::vectorDotProduct(const std::shared_ptr<Matrix<type>> A,
    const std::shared_ptr<Matrix<type>> B, cudaStream_t stream) {
    // elementwise multiply
    this->elementwiseMultiply(A, B, stream);

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
    if (this->rows != value->rows) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::sum: shape does not match");
    }

    // get minimum columns
    dtype::size columns = std::min(this->dataCols, value->dataCols);

    // kernel settings
    dim3 blocks(this->dataRows / matrix::block_size,
        columns == 1 ? 1 : columns / matrix::block_size);
    dim3 threads(matrix::block_size,
        columns == 1 ? 1 : matrix::block_size);
    dtype::size offset = 1;

    // start kernel once
    matrixKernel::sum<type>(blocks, threads, stream, value->deviceData,
        this->dataRows, offset, this->deviceData);

    // start kernel
    do {
        // update settings
        offset *= matrix::block_size;
        blocks.x = (blocks.x + matrix::block_size - 1) /
            matrix::block_size;

        matrixKernel::sum(blocks, threads, stream, this->deviceData,
            this->dataRows, offset, this->deviceData);

    }
    while (offset * matrix::block_size < this->dataRows);
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
    if (this->rows != value->rows) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::min: shape does not match");
    }

    // kernel settings
    dtype::size blocks = this->dataRows / matrix::block_size;
    dtype::size offset = 1;

    // start kernel once
    matrixKernel::min<type>(blocks, matrix::block_size, stream,
        value->deviceData, this->rows, offset, this->deviceData);

    // start kernel
    do {
        // update settings
        offset *= matrix::block_size;
        blocks = (blocks + matrix::block_size - 1) / matrix::block_size;

        matrixKernel::min<type>(blocks, matrix::block_size, stream,
            this->deviceData, this->rows, offset, this->deviceData);

    }
    while (offset * matrix::block_size < this->dataRows);
}

namespace mpFlow {
namespace numeric {
    template <>
    void Matrix<dtype::complex>::min(const std::shared_ptr<Matrix<dtype::complex>>,
    cudaStream_t) {
        throw std::logic_error("mpFlow::numeric::Matrix::min: not possible for complex values");
    }
}
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
    if (this->rows != value->rows) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::max: shape does not match");
    }

    // kernel settings
    dtype::size blocks = this->dataRows / matrix::block_size;
    dtype::size offset = 1;

    // start kernel once
    matrixKernel::max<type>(blocks, matrix::block_size, stream,
        value->deviceData, this->rows, offset, this->deviceData);

    // start kernel
    do {
        // update settings
        offset *= matrix::block_size;
        blocks = (blocks + matrix::block_size - 1) / matrix::block_size;

        matrixKernel::max<type>(blocks, matrix::block_size, stream,
            this->deviceData, this->rows, offset, this->deviceData);

    }
    while (offset * matrix::block_size < this->dataRows);
}

namespace mpFlow {
namespace numeric {
    template <>
    void Matrix<dtype::complex>::max(const std::shared_ptr<Matrix<dtype::complex>>,
    cudaStream_t) {
        throw std::logic_error("mpFlow::numeric::Matrix::max: not possible for complex values");
    }
}
}

// save matrix to stream
template <
    class type
>
void mpFlow::numeric::Matrix<type>::savetxt(std::ostream* ostream, char delimiter) const {
    // check input
    if (this->hostData == nullptr) {
        throw std::logic_error("mpFlow::numeric::Matrix::savetxt: host memory was not allocated");
    }
    if (ostream == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::savetxt: ostream == nullptr");
    }

    // write data
    for (dtype::index row = 0; row < this->rows; ++row) {
        for (dtype::index column = 0; column < this->cols - 1; ++column) {
            *ostream << (*this)(row, column) << delimiter;
        }
        *ostream << (*this)(row, this->cols - 1) << std::endl;
    }
}

// complex specialisation
namespace mpFlow {
namespace numeric {
    template <>
    void Matrix<dtype::complex>::savetxt(std::ostream* ostream, char delimiter) const {
        // check input
        if (this->hostData == nullptr) {
            throw std::logic_error("mpFlow::numeric::Matrix::savetxt: host memory was not allocated");
        }
        if (ostream == nullptr) {
            throw std::invalid_argument("mpFlow::numeric::Matrix::savetxt: ostream == nullptr");
        }

        // write data
        for (dtype::index row = 0; row < this->rows; ++row) {
            for (dtype::index column = 0; column < this->cols - 1; ++column) {
                *ostream << (*this)(row, column).real() << delimiter;
                *ostream << (*this)(row, column).imag() << delimiter;
            }
            *ostream << (*this)(row, this->cols - 1).real() << delimiter;
            *ostream << (*this)(row, this->cols - 1).imag() << std::endl;
        }
    }
}
}

// save matrix to file
template <
    class type
>
void mpFlow::numeric::Matrix<type>::savetxt(const std::string filename, char delimiter) const {
    // check input
    if (this->hostData == nullptr) {
        throw std::logic_error("mpFlow::numeric::Matrix::savetxt: host memory was not allocated");
    }

    // open file stream
    std::ofstream file(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("mpFlow::numeric::Matrix::savetxt: cannot open file!");
    }

    // save matrix
    this->savetxt(&file, delimiter);

    // close file
    file.close();
}

// load matrix from stream
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::Matrix<type>::loadtxt(std::istream* istream,
    cudaStream_t stream, char delimiter) {
    // check input
    if (istream == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::Matrix::loadtxt: istream == nullptr");
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
            std::string token;
            std::getline(line_stream, token, delimiter);

            // extract value
            std::stringstream valueStream(token);
            valueStream >> value;

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
    for (dtype::index row = 0; row < matrix->rows; ++row) {
        for (dtype::index column = 0; column < matrix->cols; ++column) {
            (*matrix)(row, column) = values[row][column];
        }
    }

    // copy data to device
    matrix->copyToDevice(stream);

    return matrix;
}

// complex specialisation
namespace mpFlow {
namespace numeric {
    template <>
    std::shared_ptr<Matrix<dtype::complex>> Matrix<dtype::complex>::loadtxt(std::istream* istream,
        cudaStream_t stream, char delimiter) {
        // load float view
        auto floatView = Matrix<dtype::real>::loadtxt(istream, stream, delimiter);

        // check shape
        if (floatView->cols % 2 != 0) {
            throw std::logic_error("mpFlow::numeric::Matrix::loadtxt: floatView->cols must be multiple of 2");
        }

        // convert to complex matrix
        auto complexMatrix = std::make_shared<Matrix<dtype::complex>>(
            floatView->rows, floatView->cols / 2, stream);

        for (dtype::index row = 0; row < complexMatrix->rows; ++row)
        for (dtype::index col = 0; col < complexMatrix->cols; ++col) {
            (*complexMatrix)(row, col) = dtype::complex((*floatView)(row, col * 2),
                (*floatView)(row, col * 2 + 1));
        }
        complexMatrix->copyToDevice(stream);

        return complexMatrix;
    }
}
}

// load matrix from file
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::Matrix<type>::loadtxt(
    const std::string filename, cudaStream_t stream, char delimiter) {
    // open file stream
    std::ifstream file(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::logic_error("mpFlow::numeric::Matrix::loadtxt: cannot open file!");
    }

    // load matrix
    auto matrix = Matrix<type>::loadtxt(&file, stream, delimiter);

    // close file
    file.close();

    return matrix;
}

// converts matrix to eigen array
template <
    class type
>
Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::matrix::toEigen(
    std::shared_ptr<Matrix<type>> matrix) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::matrix::toEigen: matrix == nullptr");
    }

    // create eigen array with mpflow_type
    Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> array(
        matrix->dataRows, matrix->dataCols);

    // copy data
    memcpy(array.data(), matrix->hostData, sizeof(type) *
        array.rows() * array.cols());

    // resize array
    array.conservativeResize(matrix->rows, matrix->cols);

    return array;
}

template <
    class type
>
Eigen::Array<std::complex<type>, Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::matrix::toEigen(
    std::shared_ptr<Matrix<thrust::complex<type>>> matrix) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::matrix::toEigen: matrix == nullptr");
    }

    // create eigen array with mpflow_type
    Eigen::Array<std::complex<type>, Eigen::Dynamic, Eigen::Dynamic> array(
        matrix->dataRows, matrix->dataCols);

    // copy data
    memcpy(array.data(), matrix->hostData, sizeof(thrust::complex<type>) *
        array.rows() * array.cols());

    // resize array
    array.conservativeResize(matrix->rows, matrix->cols);

    return array;
}

// converts eigen array to matrix
template <
    class mpflow_type,
    class eigen_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpflow_type>> mpFlow::numeric::matrix::fromEigen(
    Eigen::Ref<const Eigen::Array<eigen_type, Eigen::Dynamic, Eigen::Dynamic>> array,
    cudaStream_t stream) {
    // convert array to mpflow_type
    Eigen::Array<mpflow_type, Eigen::Dynamic, Eigen::Dynamic> mpflow_array =
        array.template cast<mpflow_type>();

    // create mpflow matrix and resize eigen array to correct size
    auto matrix = std::make_shared<Matrix<mpflow_type>>(mpflow_array.rows(),
        mpflow_array.cols(), stream);
    mpflow_array.conservativeResize(matrix->dataRows, matrix->dataCols);

    // copy data
    memcpy(matrix->hostData, mpflow_array.data(), sizeof(mpflow_type) *
        matrix->dataRows * matrix->dataCols);

    // copy data to device
    matrix->copyToDevice(stream);

    return matrix;
}

// specialisation
template Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::matrix::toEigen(
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>);
template Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::matrix::toEigen(
    std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>);
template Eigen::Array<std::complex<mpFlow::dtype::real>, Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::matrix::toEigen(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<mpFlow::dtype::real>>>);

template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> mpFlow::numeric::matrix::fromEigen(
    Eigen::Ref<const Eigen::ArrayXXf>, cudaStream_t);
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> mpFlow::numeric::matrix::fromEigen(
    Eigen::Ref<const Eigen::ArrayXXd>, cudaStream_t);
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> mpFlow::numeric::matrix::fromEigen(
    Eigen::Ref<const Eigen::ArrayXXi>, cudaStream_t);
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> mpFlow::numeric::matrix::fromEigen(
    Eigen::Ref<const Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, Eigen::Dynamic>>, cudaStream_t);
template std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> mpFlow::numeric::matrix::fromEigen(
    Eigen::Ref<const Eigen::Array<long, Eigen::Dynamic, Eigen::Dynamic>>, cudaStream_t);

template class mpFlow::numeric::Matrix<mpFlow::dtype::real>;
template class mpFlow::numeric::Matrix<mpFlow::dtype::complex>;
template class mpFlow::numeric::Matrix<mpFlow::dtype::index>;
template class mpFlow::numeric::Matrix<Eigen::ArrayXXi::Scalar>;
