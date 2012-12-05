// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/sparse_matrix.h"

// create new sparse matrix
fastEIT::SparseMatrix::SparseMatrix(const Matrix<dtype::real>& matrix, cudaStream_t stream) {
    // create empty sparse matrix
    this->init(matrix.rows(), matrix.columns(), stream);

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
    if ((this->rows() % Matrix<dtype::real>::block_size != 0) && (this->rows() != 1)) {
        this->data_rows_ = (this->rows() / Matrix<dtype::real>::block_size + 1) *
            Matrix<dtype::real>::block_size;
    }
    if ((this->columns() % Matrix<dtype::real>::block_size != 0) && (this->columns() != 1)) {
        this->data_columns_ = (this->columns() / Matrix<dtype::real>::block_size + 1) *
            Matrix<dtype::real>::block_size;
    }

    // create matrices
    if (cudaMalloc((void**)&this->values_, sizeof(dtype::real) *
        this->data_rows() * SparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("SparseMatrix::init: create memory");
    }

    if (cudaMalloc((void**)&this->column_ids_, sizeof(dtype::index) *
        this->data_rows() * SparseMatrix::block_size) != cudaSuccess) {
        throw std::logic_error("SparseMatrix::init: create memory");
    }
}

// release sparse matrix
fastEIT::SparseMatrix::~SparseMatrix() {
    // release matrices
    cudaFree(this->values_);
    cudaFree(this->column_ids_);
}

// convert to sparse matrix kernel
__global__ void sparseCreateKernel(const fastEIT::dtype::real* matrix, fastEIT::dtype::size rows,
    fastEIT::dtype::size columns, fastEIT::dtype::real* values, fastEIT::dtype::index* columnIds,
    fastEIT::dtype::index* elementCount) {
    // get id
    fastEIT::dtype::index i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    fastEIT::dtype::size count = 0;

    // init values and columnIds
    for (fastEIT::dtype::index j = 0; j < fastEIT::SparseMatrix::block_size; j++) {
        values[i * fastEIT::SparseMatrix::block_size + j] = 0.0f;
        columnIds[i * fastEIT::SparseMatrix::block_size + j] = -1;
    }

    // search non-zero elements
    fastEIT::dtype::real element = 0.0f;
    for (fastEIT::dtype::index j = 0; j < columns; j++) {
        // get element
        element = matrix[i + j * rows];

        // check for non-zero
        if (element != 0.0f) {
            values[i * fastEIT::SparseMatrix::block_size + count] = element;
            columnIds[i * fastEIT::SparseMatrix::block_size + count] = j;

            // increment count
            count++;

            // check count
            if (count >= fastEIT::SparseMatrix::block_size) {
                break;
            }
        }
    }

    // save element count
    elementCount[i] = count;
}

// convert to sparse matrix
void fastEIT::SparseMatrix::convert(const Matrix<dtype::real>& matrix, cudaStream_t stream) {
    // create elementCount matrix
    fastEIT::Matrix<dtype::index> elementCount(this->data_rows(), 1, stream);
    fastEIT::Matrix<dtype::index> maxCount(this->data_rows(), 1, stream);

    // execute kernel
    sparseCreateKernel<<<this->data_rows() / fastEIT::Matrix<dtype::real>::block_size,
        fastEIT::Matrix<dtype::real>::block_size, 0, stream>>>(matrix.device_data(), matrix.data_rows(),
        matrix.data_columns(), this->values(), this->column_ids(), elementCount.device_data());

    // get max count
    maxCount.max(elementCount, stream);
    maxCount.copyToHost(stream);
    cudaStreamSynchronize(stream);

    // save density
    this->density() = maxCount(0, 0);
}

// sparse matrix multiply kernel
__global__ void sparseMultiplyKernel(const fastEIT::dtype::real* values,
    const fastEIT::dtype::index* columnIds, const fastEIT::dtype::real* matrix, fastEIT::dtype::size rows,
    fastEIT::dtype::size columns, fastEIT::dtype::size density, fastEIT::dtype::real* result) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    fastEIT::dtype::real res = 0.0f;
    fastEIT::dtype::index id = -1;

    // read column ids to local memory
    __shared__ fastEIT::dtype::index columnId[
        fastEIT::SparseMatrix::block_size * fastEIT::SparseMatrix::block_size];
    __shared__ fastEIT::dtype::real value[
        fastEIT::SparseMatrix::block_size * fastEIT::SparseMatrix::block_size];

    columnId[threadIdx.x * fastEIT::SparseMatrix::block_size + threadIdx.y] = row < rows ?
        columnIds[row * fastEIT::SparseMatrix::block_size + threadIdx.y] : -1;
    value[threadIdx.x * fastEIT::SparseMatrix::block_size + threadIdx.y] = row < rows ?
        values[row * fastEIT::SparseMatrix::block_size + threadIdx.y] : 0.0f;

    __syncthreads();

    // check ids
    if ((row >= rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (fastEIT::dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * fastEIT::SparseMatrix::block_size + j];

         res += id != -1 ? matrix[id + column * rows] *
            value[threadIdx.x * fastEIT::SparseMatrix::block_size + j] : 0.0f;
    }

    // set result
    result[row + column * rows] = res;
}

// sparse matrix multiply
void fastEIT::SparseMatrix::multiply(const Matrix<dtype::real>& matrix, cudaStream_t stream,
    Matrix<dtype::real>* result) const {
    // check input
    if (result == NULL) {
        throw std::invalid_argument("SparseMatrix::multiply: result == NULL");
    }

    // check size
    if ((result->data_rows() != this->data_rows()) || (this->data_columns() != matrix.data_rows()) ||
        (result->data_columns() != matrix.data_columns())) {
        throw std::invalid_argument("SparseMatrix::multiply: size");
    }

    // kernel dimension
    dim3 global((result->data_rows() + block_size - 1) / block_size,
        (result->data_columns() + block_size - 1) / block_size);
    dim3 local(block_size, block_size);

    // execute kernel
    sparseMultiplyKernel<<<global, local, 0, stream>>>(this->values(), this->column_ids(),
        matrix.device_data(), result->data_rows(), result->data_columns(), this->density(),
        result->device_data());
}
