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

#include "mpflow/mpflow.h"
#include "mpflow/numeric/sparse_matrix_kernel.h"
#include <fstream>

// create new sparse matrix
template <
    class type
>
mpFlow::numeric::SparseMatrix<type>::SparseMatrix(std::shared_ptr<Matrix<type> const> const matrix,
    cudaStream_t const stream)
    : rows(matrix->rows), cols(matrix->cols) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::SparseMatrix: matrix == nullptr");
    }

    // create empty sparse matrix
    this->init(matrix->rows, matrix->cols, stream);

    // convert to sparse_matrix
    this->convert(matrix, stream);
}

// create empty sparse matrix
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::init(unsigned const rows, unsigned const columns,
    cudaStream_t const stream) {
    // check input
    if (rows == 0) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::init: rows == 0");
    }
    if (columns == 0) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::init: columns == 0");
    }

    // init struct
    this->dataRows = rows;
    this->dataCols = columns;
    this->density = 0;

    // correct size to block size
    if ((this->rows % matrix::blockSize != 0) && (this->rows != 1)) {
        this->dataRows = math::roundTo(this->rows, matrix::blockSize);
    }
    if ((this->cols % matrix::blockSize != 0) && (this->cols != 1)) {
        this->dataCols = math::roundTo(this->cols, matrix::blockSize);
    }

    // create matrix device data memory
    cudaError_t error = cudaSuccess;
    error = cudaMalloc((void**)&this->deviceValues, sizeof(type) *
        this->dataRows * sparseMatrix::blockSize);

    CudaCheckError();
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::SparseMatrix::init: create device data memory");
    }

    error = cudaMalloc((void**)&this->deviceColumnIds, sizeof(unsigned) *
        this->dataRows * sparseMatrix::blockSize);

    CudaCheckError();
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::SparseMatrix::init: create device data memory");
    }

    // create matrix host data memory
    error = cudaHostAlloc((void**)&this->hostValues, sizeof(type) *
        this->dataRows * sparseMatrix::blockSize, cudaHostAllocDefault);

    CudaCheckError();
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::SparseMatrix::init: create host data memory");
    }

    error = cudaHostAlloc((void**)&this->hostColumnIds, sizeof(unsigned) *
        this->dataRows * sparseMatrix::blockSize, cudaHostAllocDefault);

    CudaCheckError();
    if (error != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::SparseMatrix::init: create host data memory");
    }

    // fill matrix with default data
    for (unsigned row = 0; row < this->dataRows; ++row)
    for (unsigned col = 0; col < sparseMatrix::blockSize; ++col) {
        this->hostValues[row * sparseMatrix::blockSize + col] = 0.0f;
        this->hostColumnIds[row * sparseMatrix::blockSize + col] = constants::invalidIndex;
    }
    this->copyToDevice(stream);
    cudaStreamSynchronize(stream);
}

// release sparse matrix
template <
    class type
>
mpFlow::numeric::SparseMatrix<type>::~SparseMatrix() {
    // release matrices
    cudaFree(this->deviceValues);
    cudaFree(this->deviceColumnIds);
    cudaFreeHost(this->hostValues);
    cudaFreeHost(this->hostColumnIds);
    CudaCheckError();
}

// copy matrix
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::copy(std::shared_ptr<SparseMatrix<type> const> const other,
    cudaStream_t const stream) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::SparseMatrix::copy: other == nullptr");
    }

    // check size
    if ((other->rows != this->rows) ||
        (other->cols != this->cols)) {
        throw std::invalid_argument(
            str::format("mpFlow::numeric::SparseMatrix::copy: shape does not match: (%d, %d) != (%d, %d)")
            (this->rows, this->cols, other->rows, other->cols));
    }

    // copy data
    this->density = other->density;
    CudaSafeCall(
        cudaMemcpyAsync(this->deviceValues, other->deviceValues,
            sizeof(type) * this->dataRows * sparseMatrix::blockSize,
            cudaMemcpyDeviceToDevice, stream));
    CudaSafeCall(
        cudaMemcpyAsync(this->deviceColumnIds, other->deviceColumnIds,
            sizeof(unsigned) * this->dataRows * sparseMatrix::blockSize,
            cudaMemcpyDeviceToDevice, stream));

    CudaCheckError();
}

// copy to device
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::copyToDevice(cudaStream_t const stream) {
    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->deviceValues, this->hostValues,
            sizeof(type) * this->dataRows * sparseMatrix::blockSize,
            cudaMemcpyHostToDevice, stream));
    CudaSafeCall(
        cudaMemcpyAsync(this->deviceColumnIds, this->hostColumnIds,
            sizeof(unsigned) * this->dataRows * sparseMatrix::blockSize,
            cudaMemcpyHostToDevice, stream));

    CudaCheckError();
}

// copy to host
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::copyToHost(cudaStream_t const stream) {
    // copy host buffer to device
    CudaSafeCall(
        cudaMemcpyAsync(this->hostValues, this->deviceValues,
            sizeof(type) * this->dataRows * sparseMatrix::blockSize,
            cudaMemcpyDeviceToHost, stream));
    CudaSafeCall(
        cudaMemcpyAsync(this->hostColumnIds, this->deviceColumnIds,
            sizeof(unsigned) * this->dataRows * sparseMatrix::blockSize,
            cudaMemcpyDeviceToHost, stream));

    CudaCheckError();
}

// convert to sparse matrix
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::convert(std::shared_ptr<Matrix<type> const> const matrix,
    cudaStream_t const stream) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::convert: matrix == nullptr");
    }

    // create elementCount matrix
    auto elementCount = std::make_shared<Matrix<unsigned>>(this->dataRows, 1, stream);
    auto maxCount = std::make_shared<Matrix<unsigned>>(this->dataRows, 1, stream);

    // execute kernel
    sparseMatrixKernel::convert(this->dataRows / matrix::blockSize,
        matrix::blockSize, stream, matrix->deviceData, matrix->dataRows,
        matrix->dataCols, this->deviceValues, this->deviceColumnIds,
        elementCount->deviceData);

    // get max count
    maxCount->max(elementCount, stream);
    maxCount->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // save density
    this->density = (*maxCount)(0, 0);
}

// convert to matrix
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::SparseMatrix<type>::toMatrix(
    cudaStream_t const stream) const {
    // create empty matrix
    auto matrix = std::make_shared<Matrix<type>>(this->rows,
        this->cols, stream);

    // convert to matrix
    sparseMatrixKernel::convertToMatrix(this->dataRows / matrix::blockSize,
        matrix::blockSize, stream, this->deviceValues, this->deviceColumnIds,
        this->density, matrix->dataRows, matrix->deviceData);

    return matrix;
}

// scalar multiply
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::scalarMultiply(type const scalar, cudaStream_t const stream) {
    // kernel dimensions
    dim3 blocks(this->dataRows / matrix::blockSize, sparseMatrix::blockSize);
    dim3 threads(matrix::blockSize, 1);
    
    sparseMatrixKernel::scalarMultiply(blocks, threads, stream, scalar, this->deviceValues);
}

// sparse matrix multiply
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::multiply(std::shared_ptr<Matrix<type> const> const matrix,
    cudaStream_t const stream, std::shared_ptr<Matrix<type>> result) const {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::multiply: matrix == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::multiply: result == nullptr");
    }

    // check size
    if ((result->dataRows != this->dataRows) ||
        (this->dataCols != matrix->dataRows) ||
        (result->dataCols != matrix->dataCols)) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::multiply: shape does not match");
    }

    // kernel dimension
    dim3 blocks((result->dataRows + sparseMatrix::blockSize - 1) / sparseMatrix::blockSize,
        (result->dataCols + sparseMatrix::blockSize - 1) / sparseMatrix::blockSize);
    dim3 threads(sparseMatrix::blockSize, sparseMatrix::blockSize);

    // execute kernel
    sparseMatrixKernel::multiply(blocks, threads, stream, this->deviceValues, this->deviceColumnIds,
        matrix->deviceData, result->dataRows, matrix->dataRows,
        result->dataCols, this->density, result->deviceData);
}

// converts matrix to eigen array
template <
    class type
>
Eigen::Array<typename mpFlow::typeTraits::convertComplexType<type>::type,
    Eigen::Dynamic, Eigen::Dynamic> mpFlow::numeric::SparseMatrix<type>::toEigen(cudaStream_t const stream) const {
    // convert sparse matrix to full matrix
    auto const fullMatrix = this->toMatrix(stream);
    fullMatrix->copyToHost(stream);
    cudaStreamSynchronize(stream);
    
    // convert full matrix to eigen array
    auto const array = fullMatrix->toEigen();
    
    return array;
}

// save matrix to stream
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::savetxt(std::ostream& ostream, char const delimiter) const {
    // write data
    for (unsigned row = 0; row < this->rows; ++row) {
        for (unsigned col = 0; col < this->cols - 1; ++col) {
            ostream << this->getValue(row, col) << delimiter;
        }
        ostream << this->getValue(row, this->cols - 1);

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
void mpFlow::numeric::SparseMatrix<type>::savetxt(std::string const filename, char const delimiter) const {
    // open file stream
    std::ofstream file(filename.c_str());

    // check open
    if (file.fail()) {
        throw std::runtime_error(
            str::format("mpFlow::numeric::SparseMatrix::savetxt: cannot open file: %s")(filename));
    }

    // save matrix
    file.precision(std::numeric_limits<double>::digits10 + 1);
    this->savetxt(file, delimiter);

    // close file
    file.close();
}

// accessors
template <
    class type
>
unsigned mpFlow::numeric::SparseMatrix<type>::getColumnId(unsigned const row, unsigned const col) const {
    // check index sizes
    if ((row >= this->rows) || (col >= this->cols)) {
        throw std::logic_error(
            str::format("mpFlow::numeric::SparseMatrix::getColumnId(): index out of range: (%d, %d) >= (%d, %d)")
                (row, col, this->rows, this->cols));
    }

    // find index in column ids
    unsigned columnId = constants::invalidIndex;
    for (unsigned i = 0; i < sparseMatrix::blockSize; ++i) {
        if (this->hostColumnIds[row * sparseMatrix::blockSize + i] == col) {
            columnId = i;
            break;
        }
    }

    return columnId;
}

template <
    class type
>
type mpFlow::numeric::SparseMatrix<type>::getValue(unsigned const row, unsigned const col) const {
    // get column id
    unsigned columnId = this->getColumnId(row, col);

    if (columnId == constants::invalidIndex) {
        return type(0);
    }
    else {
        return this->hostValues[row * sparseMatrix::blockSize + columnId];
    }
}

template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::setValue(unsigned const row, unsigned const col, type const& value) {
    // check index sizes
    if ((row >= this->rows) || (col >= this->cols)) {
        throw std::logic_error(
            str::format("mpFlow::numeric::SparseMatrix::setValue(): index out of range: (%d, %d) >= (%d, %d)")
                (row, col, this->rows, this->cols));
    }

    // find index in column ids
    unsigned columnId = constants::invalidIndex;
    for (unsigned i = 0; i < sparseMatrix::blockSize; ++i) {
        if (this->hostColumnIds[row * sparseMatrix::blockSize + i] == col) {
            columnId = i;
            break;
        }
        else if (this->hostColumnIds[row * sparseMatrix::blockSize + i] == constants::invalidIndex) {
            columnId = i;
            this->hostColumnIds[row * sparseMatrix::blockSize + i] = col;
            this->density = std::max(this->density, i + 1);
            break;
        }
        else if (i == sparseMatrix::blockSize - 1) {
            throw std::logic_error("mpFlow::numeric::SparseMatrix::operator(): sparse format full");
        }
    }

    this->hostValues[row * sparseMatrix::blockSize + columnId] = value;
}

// specialisations
template class mpFlow::numeric::SparseMatrix<float>;
template class mpFlow::numeric::SparseMatrix<double>;
template class mpFlow::numeric::SparseMatrix<thrust::complex<float>>;
template class mpFlow::numeric::SparseMatrix<thrust::complex<double>>;
template class mpFlow::numeric::SparseMatrix<unsigned>;
template class mpFlow::numeric::SparseMatrix<int>;
