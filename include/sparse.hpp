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
    SparseMatrix(dtype::size rows, dtype::size columns, cudaStream_t stream = NULL);
    virtual ~SparseMatrix();

public:
    // block size
    static const dtype::size blockSize = 16;

// accessors
public:
    dtype::real* hostData() const { return this->mHostData; }
    dtype::real* deviceData() const { return this->mDeviceData; }
    dtype::size rows() const { return this->mRows; }
    dtype::size columns() const { return this->mColumns; }

// member
private:
    dtype::real* mHostData;
    dtype::real* mDeviceData;
    dtype::size mRows;
    dtype::size mColumns;
};
#endif
