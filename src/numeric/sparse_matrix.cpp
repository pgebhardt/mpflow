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

// create new sparse matrix
template <
    class type
>
mpFlow::numeric::SparseMatrix<type>::SparseMatrix(const std::shared_ptr<Matrix<type>> matrix,
    cudaStream_t stream) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::SparseMatrix: matrix == nullptr");
    }

    // create empty sparse matrix
    this->init(matrix->rows, matrix->cols);

    // convert to sparse_matrix
    this->convert(matrix, stream);
}

// create empty sparse matrix
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::init(dtype::size rows, dtype::size columns) {
    // check input
    if (rows == 0) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::init: rows == 0");
    }
    if (columns == 0) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::init: columns == 0");
    }

    // init struct
    this->rows = rows;
    this->cols = columns;
    this->dataRows = rows;
    this->dataCols = columns;
    this->density = 0;

    // correct size to block size
    if ((this->rows % matrix::block_size != 0) && (this->rows != 1)) {
        this->dataRows = math::roundTo(this->rows, matrix::block_size);
    }
    if ((this->cols % matrix::block_size != 0) && (this->cols != 1)) {
        this->dataCols = math::roundTo(this->cols, matrix::block_size);
    }

    // create matrices
    if (cudaMalloc((void**)&this->values, sizeof(type) *
        this->dataRows * sparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::numeric::SparseMatrix::init: create memory");
    }

    if (cudaMalloc((void**)&this->columnIds, sizeof(dtype::index) *
        this->dataRows * sparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::numeric::SparseMatrix::init: create memory");
    }
}

// release sparse matrix
template <
    class type
>
mpFlow::numeric::SparseMatrix<type>::~SparseMatrix() {
    // release matrices
    cudaFree(this->values);
    cudaFree(this->columnIds);
    CudaCheckError();
}

// convert to sparse matrix
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::convert(const std::shared_ptr<Matrix<type>> matrix,
    cudaStream_t stream) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::convert: matrix == nullptr");
    }

    // create elementCount matrix
    auto elementCount = std::make_shared<Matrix<dtype::index>>(this->dataRows, 1, stream);
    auto maxCount = std::make_shared<Matrix<dtype::index>>(this->dataRows, 1, stream);

    // execute kernel
    sparseMatrixKernel::convert(this->dataRows / matrix::block_size,
        matrix::block_size, stream, matrix->deviceData, matrix->dataRows,
        matrix->dataCols, this->values, this->columnIds,
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
    cudaStream_t stream) {
    // create empty matrix
    auto matrix = std::make_shared<Matrix<type>>(this->rows,
        this->cols, stream);

    // convert to matrix
    sparseMatrixKernel::convertToMatrix(this->dataRows / matrix::block_size,
        matrix::block_size, stream, this->values, this->columnIds,
        this->density, matrix->dataRows, matrix->deviceData);

    return matrix;
}

// sparse matrix multiply
template <
    class type
>
void mpFlow::numeric::SparseMatrix<type>::multiply(const std::shared_ptr<Matrix<type>> matrix,
    cudaStream_t stream, std::shared_ptr<Matrix<type>> result) const {
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
    dim3 blocks((result->dataRows + sparseMatrix::block_size - 1) / sparseMatrix::block_size,
        (result->dataCols + sparseMatrix::block_size - 1) / sparseMatrix::block_size);
    dim3 threads(sparseMatrix::block_size, sparseMatrix::block_size);

    // execute kernel
    sparseMatrixKernel::multiply(blocks, threads, stream, this->values, this->columnIds,
        matrix->deviceData, result->dataRows, matrix->dataRows,
        result->dataCols, this->density, result->deviceData);
}

// specialisations
template class mpFlow::numeric::SparseMatrix<mpFlow::dtype::real>;
template class mpFlow::numeric::SparseMatrix<mpFlow::dtype::complex>;
template class mpFlow::numeric::SparseMatrix<mpFlow::dtype::index>;
template class mpFlow::numeric::SparseMatrix<Eigen::ArrayXXi::Scalar>;
