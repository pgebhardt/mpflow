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
    this->init(matrix->rows(), matrix->columns());

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
    this->rows_ = rows;
    this->columns_ = columns;
    this->data_rows_ = rows;
    this->data_columns_ = columns;
    this->density_ = 0;
    this->values_ = nullptr;
    this->column_ids_ = nullptr;

    // correct size to block size
    if ((this->rows() % matrix::block_size != 0) && (this->rows() != 1)) {
        this->data_rows_ = math::roundTo(this->rows(), matrix::block_size);
    }
    if ((this->columns() % matrix::block_size != 0) && (this->columns() != 1)) {
        this->data_columns_ = math::roundTo(this->columns(), matrix::block_size);
    }


    // create matrices
    if (cudaMalloc((void**)&this->values_, sizeof(type) *
        this->data_rows() * sparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::numeric::SparseMatrix::init: create memory");
    }

    if (cudaMalloc((void**)&this->column_ids_, sizeof(dtype::index) *
        this->data_rows() * sparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("mpFlow::numeric::numeric::SparseMatrix::init: create memory");
    }
}

// release sparse matrix
template <
    class type
>
mpFlow::numeric::SparseMatrix<type>::~SparseMatrix() {
    // release matrices
    cudaFree(this->values_);
    cudaFree(this->column_ids_);
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
    auto elementCount = std::make_shared<Matrix<dtype::index>>(this->data_rows(), 1, stream);
    auto maxCount = std::make_shared<Matrix<dtype::index>>(this->data_rows(), 1, stream);

    // execute kernel
    sparseMatrixKernel::convert(this->data_rows() / matrix::block_size,
        matrix::block_size, stream, matrix->device_data(), matrix->data_rows(),
        matrix->data_columns(), this->values(), this->column_ids(),
        elementCount->device_data());

    // get max count
    maxCount->max(elementCount, stream);
    maxCount->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // save density
    this->density() = (*maxCount)(0, 0);
}

// convert to matrix
template <
    class type
>
std::shared_ptr<mpFlow::numeric::Matrix<type>> mpFlow::numeric::SparseMatrix<type>::toMatrix(
    cudaStream_t stream) {
    // create empty matrix
    auto matrix = std::make_shared<Matrix<type>>(this->rows(),
        this->columns(), stream);

    // convert to matrix
    sparseMatrixKernel::convertToMatrix(this->data_rows() / matrix::block_size,
        matrix::block_size, stream, this->values(), this->column_ids(),
        this->density(), matrix->data_rows(), matrix->device_data());

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
    if ((result->data_rows() != this->data_rows()) ||
        (this->data_columns() != matrix->data_rows()) ||
        (result->data_columns() != matrix->data_columns())) {
        throw std::invalid_argument("mpFlow::numeric::numeric::SparseMatrix::multiply: shape does not match");
    }

    // kernel dimension
    dim3 blocks((result->data_rows() + sparseMatrix::block_size - 1) / sparseMatrix::block_size,
        (result->data_columns() + sparseMatrix::block_size - 1) / sparseMatrix::block_size);
    dim3 threads(sparseMatrix::block_size, sparseMatrix::block_size);

    // execute kernel
    sparseMatrixKernel::multiply(blocks, threads, stream, this->values(), this->column_ids(),
        matrix->device_data(), result->data_rows(), matrix->data_rows(),
        result->data_columns(), this->density(), result->device_data());
}

// specialisations
template class mpFlow::numeric::SparseMatrix<mpFlow::dtype::real>;
template class mpFlow::numeric::SparseMatrix<mpFlow::dtype::index>;
