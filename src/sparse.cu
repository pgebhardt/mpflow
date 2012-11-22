// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create new sparse matrix
SparseMatrix::SparseMatrix(Matrix<dtype::real>* matrix, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("SparseMatrix::SparseMatrix: matrix == NULL");
    }

    // create empty sparse matrix
    this->init(matrix->rows(), matrix->columns(), stream);

    // convert to sparse_matrix
    this->convert(matrix, stream);
}

// create empty sparse matrix
void SparseMatrix::init(dtype::size rows, dtype::size columns, cudaStream_t stream) {
    // check input
    if (rows == 0) {
        throw invalid_argument("SparseMatrix::init: rows == 0");
    }
    if (columns == 0) {
        throw invalid_argument("SparseMatrix::init: columns == 0");
    }

    // init struct
    this->mRows = rows;
    this->mColumns = columns;
    this->mDensity = 0;
    this->mValues = NULL;
    this->mColumnIds = NULL;

    // correct size to block size
    if ((this->rows() % Matrix<dtype::real>::blockSize != 0) && (this->rows() != 1)) {
        this->mRows = (this->rows() / Matrix<dtype::real>::blockSize + 1) *
            Matrix<dtype::real>::blockSize;
    }
    if ((this->columns() % Matrix<dtype::real>::blockSize != 0) && (this->columns() != 1)) {
        this->mColumns = (this->columns() / Matrix<dtype::real>::blockSize + 1) *
            Matrix<dtype::real>::blockSize;
    }

    // create matrices
    if (cudaMalloc((void**)&this->mValues, sizeof(dtype::real) *
        this->rows() * SparseMatrix::blockSize) != cudaSuccess) {
        throw logic_error("SparseMatrix::init: create memory");
    }

    if (cudaMalloc((void**)&this->mColumnIds, sizeof(dtype::real) *
        this->rows() * SparseMatrix::blockSize) != cudaSuccess) {
        throw logic_error("SparseMatrix::init: create memory");
    }
}

// release sparse matrix
SparseMatrix::~SparseMatrix() {
    // release matrices
    cudaFree(this->values());
    cudaFree(this->columnIds());
}

// convert to sparse matrix kernel
__global__ void sparseCreateKernel(dtype::real* values,
    dtype::index* columnIds, dtype::real* matrix,
    dtype::index* elementCount, dtype::size rows, dtype::size columns) {
    // get id
    dtype::index i = blockIdx.x * blockDim.x + threadIdx.x;

    // element count
    dtype::size count = 0;

    // init values and columnIds
    for (dtype::index j = 0; j < SparseMatrix::blockSize; j++) {
        values[i * SparseMatrix::blockSize + j] = 0.0f;
        columnIds[i * SparseMatrix::blockSize + j] = -1;
    }

    // search non-zero elements
    dtype::real element = 0.0f;
    for (dtype::index j = 0; j < columns; j++) {
        // get element
        element = matrix[i + j * rows];

        // check for non-zero
        if (element != 0.0f) {
            values[i * SparseMatrix::blockSize + count] = element;
            columnIds[i * SparseMatrix::blockSize + count] = j;

            // increment count
            count++;

            // check count
            if (count >= SparseMatrix::blockSize) {
                break;
            }
        }
    }

    // save element count
    elementCount[i] = count;
}

// convert to sparse matrix
void SparseMatrix::convert(Matrix<dtype::real>* matrix, cudaStream_t stream) {
    // check input
    if (matrix == NULL) {
        throw invalid_argument("SparseMatrix::convert: matrix == NULL");
    }

    // create elementCount matrix
    Matrix<dtype::index> elementCount(this->rows(), 1, stream);
    Matrix<dtype::index> maxCount(this->rows(), 1, stream);

    // execute kernel
    sparseCreateKernel<<<this->rows() / SparseMatrix::blockSize,
        SparseMatrix::blockSize, 0, stream>>>(
        this->values(), this->columnIds(), matrix->deviceData(),
        elementCount.deviceData(), matrix->rows(), matrix->columns());

    // get max count
    maxCount.max(&elementCount, maxCount.rows(), stream);
    maxCount.copyToHost(stream);
    cudaStreamSynchronize(stream);

    // save density
    this->mDensity = maxCount.hostData()[0];
}

// sparse matrix multiply kernel
__global__ void sparseMultiplyKernel(dtype::real* result,
    dtype::real* values, dtype::index* columnIds,
    dtype::real* matrix, dtype::size rows, dtype::size columns,
    dtype::size density) {
    // get ids
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    dtype::real res = 0.0f;
    dtype::index id = -1;

    // read column ids to local memory
    __shared__ dtype::index columnId[SparseMatrix::blockSize * SparseMatrix::blockSize];
    __shared__ dtype::real value[SparseMatrix::blockSize * SparseMatrix::blockSize];
    columnId[threadIdx.x * SparseMatrix::blockSize + threadIdx.y] = row < rows ?
        columnIds[row * SparseMatrix::blockSize + threadIdx.y] : -1;
    value[threadIdx.x * SparseMatrix::blockSize + threadIdx.y] = row < rows ?
        values[row * SparseMatrix::blockSize + threadIdx.y] : 0.0f;
    __syncthreads();

    // check ids
    if ((row >= rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (dtype::index j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * SparseMatrix::blockSize + j];

         res += id != -1 ? matrix[id + column * rows] *
            value[threadIdx.x * SparseMatrix::blockSize + j] : 0.0f;
    }

    // set result
    result[row + column * rows] = res;
}

// sparse matrix multiply
void SparseMatrix::multiply(Matrix<dtype::real>* result, Matrix<dtype::real>* matrix,
    cudaStream_t stream) {
    // check input
    if (result == NULL) {
        throw invalid_argument("SparseMatrix::multiply: result == NULL");
    }
    if (matrix == NULL) {
        throw invalid_argument("SparseMatrix::multiply: matrix == NULL");
    }

    // check size
    if ((result->rows() != this->rows()) || (this->columns() != matrix->rows()) ||
        (result->columns() != matrix->columns())) {
        throw invalid_argument("SparseMatrix::multiply: size");
    }

    // kernel dimension
    dim3 global((result->rows() + SparseMatrix::blockSize - 1) / SparseMatrix::blockSize,
        (result->columns() + SparseMatrix::blockSize - 1) / SparseMatrix::blockSize);
    dim3 local(SparseMatrix::blockSize, SparseMatrix::blockSize);

    // execute kernel
    sparseMultiplyKernel<<<global, local, 0, stream>>>(result->deviceData(), this->values(),
        this->columnIds(), matrix->deviceData(), result->rows(), result->columns(), this->density());
}
