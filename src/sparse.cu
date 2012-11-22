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
__global__ void sparse_create_kernel(dtype::real* values,
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
    sparse_create_kernel<<<this->rows() / SparseMatrix::blockSize,
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
/*
// sparse matrix multiply kernel
__global__ void sparse_multiply_kernel(linalgcuMatrixData_t* result,
    linalgcuMatrixData_t* values, linalgcuColumnId_t* columnIds,
    linalgcuMatrixData_t* matrix, linalgcuSize_t rows, linalgcuSize_t columns,
    linalgcuSize_t density) {
    // get ids
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc result
    linalgcuMatrixData_t res = 0.0f;
    linalgcuColumnId_t id = -1;

    // read column ids to local memory
    __shared__ linalgcuColumnId_t columnId[LINALGCU_SPARSE_SIZE * LINALGCU_SPARSE_SIZE];
    __shared__ linalgcuMatrixData_t value[LINALGCU_SPARSE_SIZE * LINALGCU_SPARSE_SIZE];
    columnId[threadIdx.x * LINALGCU_SPARSE_SIZE + threadIdx.y] = row < rows ?
        columnIds[row * LINALGCU_SPARSE_SIZE + threadIdx.y] : -1;
    value[threadIdx.x * LINALGCU_SPARSE_SIZE + threadIdx.y] = row < rows ?
        values[row * LINALGCU_SPARSE_SIZE + threadIdx.y] : 0.0f;
    __syncthreads();

    // check ids
    if ((row >= rows) || (column >= columns)) {
        return;
    }

    // read matrix to local memory
    for (linalgcuSize_t j = 0; j < density; j++) {
        // get column id
        id = columnId[threadIdx.x * LINALGCU_SPARSE_SIZE + j];

         res += id != -1 ? matrix[id + column * rows] *
            value[threadIdx.x * LINALGCU_SPARSE_SIZE + j] : 0.0f;
    }

    // set result
    result[row + column * rows] = res;
}

// sparse matrix multiply
extern "C"
linalgcuError_t linalgcu_sparse_matrix_multiply(linalgcuMatrix_t result,
    linalgcuSparseMatrix_t sparse, linalgcuMatrix_t matrix, cudaStream_t stream) {
    // check input
    if ((result == NULL) || (sparse == NULL) || (matrix == NULL)) {
        return LINALGCU_ERROR;
    }

    // check size
    if ((result->rows != sparse->rows) || (sparse->columns != matrix->rows) ||
        (result->columns != matrix->columns)) {
        return LINALGCU_ERROR;
    }

    // kernel dimension
    dim3 global((result->rows + LINALGCU_SPARSE_SIZE - 1) / LINALGCU_SPARSE_SIZE,
        (result->columns + LINALGCU_SPARSE_SIZE - 1) / LINALGCU_SPARSE_SIZE);
    dim3 local(LINALGCU_SPARSE_SIZE, LINALGCU_SPARSE_SIZE);

    // execute kernel
    sparse_multiply_kernel<<<global, local, 0, stream>>>(result->deviceData, sparse->values,
        sparse->columnIds, matrix->deviceData, result->rows, result->columns, sparse->density);

    return LINALGCU_SUCCESS;
}*/
