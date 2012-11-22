// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SPARSE_HPP
#define FASTEIT_SPARSE_HPP

// sparse matrix class definition
class SparseMatrix {
public:
    // constructor and destructor
    SparseMatrix(dtype::size rows, dtype::size columns, cudaStream_t stream=NULL) {
        this->init(rows, columns, stream);
    }

    SparseMatrix(Matrix<dtype::real>* matrix, cudaStream_t stream=NULL);
    virtual ~SparseMatrix();

private:
    // init empty sparse matrix
    void init(dtype::size rows, dtype::size columns, cudaStream_t stream=NULL);

    // convert to sparse matrix
    void convert(Matrix<dtype::real>* matrix, cudaStream_t stream=NULL);

public:
    // matrix multiply
    void multiply(Matrix<dtype::real>* result, Matrix<dtype::real>* matrix, cudaStream_t stream=NULL);

public:
    // block size
    static const dtype::size blockSize = 32;

// accessors
public:
    dtype::size rows() const { return this->mRows; }
    dtype::size columns() const { return this->mColumns; }
    dtype::size density() const { return this->mDensity; }
    dtype::real* values() const { return this->mValues; }
    dtype::index* columnIds() const { return this->mColumnIds; }

// member
private:
    dtype::size mRows;
    dtype::size mColumns;
    dtype::size mDensity;
    dtype::real* mValues;
    dtype::index* mColumnIds;
};
#endif
