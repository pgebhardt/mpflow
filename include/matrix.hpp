// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_MATRIX_HPP
#define FASTEIT_MATRIX_HPP

// matrix class definition
template <class type>
class Matrix {
public:
    // constructor and destructor
    Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream=NULL);
    virtual ~Matrix();

public:
    // copy methods
    void copy(Matrix<type>* other, cudaStream_t stream=NULL);
    void copyToDevice(cudaStream_t stream=NULL);
    void copyToHost(cudaStream_t stream=NULL);

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
