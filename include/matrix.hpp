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

// mathematical methods
public:
    void add(Matrix<type>* value, cudaStream_t stream=NULL);
    void multiply(Matrix<type>* A, Matrix<type>* B, cublasHandle_t handle, cudaStream_t stream=NULL);
    void scalarMultiply(type scalar, cudaStream_t stream=NULL);
    void vectorDotProduct(Matrix<type>* A, Matrix<type>* B, cudaStream_t stream=NULL);

// reduce methods
public:
    void sum(Matrix<type>* value, cudaStream_t stream=NULL);
    void min(Matrix<type>* value, dtype::size maxIndex, cudaStream_t stream=NULL);
    void max(Matrix<type>* value, dtype::size maxIndex, cudaStream_t stream=NULL);

// accessors
public:
    type* hostData() const { return this->mHostData; }
    type* deviceData() const { return this->mDeviceData; }
    dtype::size rows() const { return this->mRows; }
    dtype::size columns() const { return this->mColumns; }
    type& operator() (dtype::index i, dtype::index j) { return this->mHostData[i + j * this->rows()]; }

// member
private:
    type* mHostData;
    type* mDeviceData;
    dtype::size mRows;
    dtype::size mColumns;
};

#endif
