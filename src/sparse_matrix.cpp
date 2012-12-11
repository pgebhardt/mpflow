// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"
#include "../include/sparse_matrix_kernel.h"

// create new sparse matrix
fastEIT::SparseMatrix::SparseMatrix(const std::shared_ptr<Matrix<dtype::real>> matrix,
    cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw std::invalid_argument("SparseMatrix::SparseMatrix: matrix == NULL");
    }

    // create empty sparse matrix
    this->init(matrix->rows(), matrix->columns(), stream);

    // convert to sparse_matrix
    this->convert(matrix, stream);
}

// create empty sparse matrix
void fastEIT::SparseMatrix::init(dtype::size rows, dtype::size columns, cudaStream_t stream) {
    // check input
    if (rows == 0) {
        throw std::invalid_argument("SparseMatrix::init: rows == 0");
    }
    if (columns == 0) {
        throw std::invalid_argument("SparseMatrix::init: columns == 0");
    }

    // init struct
    this->rows_ = rows;
    this->columns_ = columns;
    this->data_rows_ = rows;
    this->data_columns_ = columns;
    this->density_ = 0;
    this->values_ = NULL;
    this->column_ids_ = NULL;

    // correct size to block size
    if ((this->rows() % matrix::block_size != 0) && (this->rows() != 1)) {
        this->data_rows_ = (this->rows() / matrix::block_size + 1) *
            matrix::block_size;
    }
    if ((this->columns() % matrix::block_size != 0) && (this->columns() != 1)) {
        this->data_columns_ = (this->columns() / matrix::block_size + 1) *
            matrix::block_size;
    }

    // create matrices
    if (cudaMalloc((void**)&this->values_, sizeof(dtype::real) *
        this->data_rows() * sparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("SparseMatrix::init: create memory");
    }

    if (cudaMalloc((void**)&this->column_ids_, sizeof(dtype::index) *
        this->data_rows() * sparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("SparseMatrix::init: create memory");
    }
}

// release sparse matrix
fastEIT::SparseMatrix::~SparseMatrix() {
    // release matrices
    cudaFree(this->values_);
    cudaFree(this->column_ids_);
}

// convert to sparse matrix
void fastEIT::SparseMatrix::convert(const std::shared_ptr<Matrix<dtype::real>> matrix,
    cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw std::invalid_argument("SparseMatrix::convert: matrix == NULL");
    }

    // create elementCount matrix
    auto elementCount = std::make_shared<Matrix<dtype::index>>(this->data_rows(), 1, stream);
    auto maxCount = std::make_shared<Matrix<dtype::index>>(this->data_rows(), 1, stream);

    // execute kernel
    sparseMatrixKernel::convert(this->data_rows() / fastEIT::matrix::block_size,
        fastEIT::matrix::block_size, stream, matrix->device_data(), matrix->data_rows(),
        matrix->data_columns(), this->values(), this->column_ids(),
        elementCount->device_data());

    // get max count
    maxCount->max(elementCount, stream);
    maxCount->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // save density
    this->density() = (*maxCount)(0, 0);
}

// sparse matrix multiply
void fastEIT::SparseMatrix::multiply(const std::shared_ptr<Matrix<dtype::real>> matrix,
    cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> result) const {
    // check input
    if (matrix == NULL) {
        throw std::invalid_argument("SparseMatrix::multiply: matrix == NULL");
    }
    if (result == NULL) {
        throw std::invalid_argument("SparseMatrix::multiply: result == NULL");
    }

    // check size
    if ((result->data_rows() != this->data_rows()) ||
        (this->data_columns() != matrix->data_rows()) ||
        (result->data_columns() != matrix->data_columns())) {
        throw std::invalid_argument("SparseMatrix::multiply: size");
    }

    // kernel dimension
    dim3 blocks((result->data_rows() + sparseMatrix::block_size - 1) / sparseMatrix::block_size,
        (result->data_columns() + sparseMatrix::block_size - 1) / sparseMatrix::block_size);
    dim3 threads(sparseMatrix::block_size, sparseMatrix::block_size);

    // execute kernel
    sparseMatrixKernel::multiply(blocks, threads, stream, this->values(), this->column_ids(),
        matrix->device_data(), result->data_rows(), result->data_columns(), this->density(),
        result->device_data());
}
