// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_CONJUGATE_HPP
#define FASTEIT_CONJUGATE_HPP

// conjugate class definition
class Conjugate {
// constructor and destructor
public:
    Conjugate(dtype::size rows, cublasHandle_t handle, cudaStream_t stream);
    virtual ~Conjugate();

public:
    // solve system
    void solve(Matrix<dtype::real>* A, Matrix<dtype::real>* x, Matrix<dtype::real>* f,
        dtype::size iterations, cublasHandle_t handle, cudaStream_t stream);

// helper methods
public:
    static void addScalar(Matrix<dtype::real>* vector, Matrix<dtype::real>* scalar,
        dtype::size rows, dtype::size columns, cudaStream_t stream);
    static void updateVector(Matrix<dtype::real>* result, Matrix<dtype::real>* x1,
        dtype::real sign, Matrix<dtype::real>* x2, Matrix<dtype::real>* r1,
        Matrix<dtype::real>* r2, cudaStream_t stream);
    static void gemv(Matrix<dtype::real>* result, Matrix<dtype::real>* matrix,
        Matrix<dtype::real>* vector, cudaStream_t);

// accessors
public:
    dtype::size rows() const { return this->mRows; }

// member
private:
    dtype::size mRows;
    Matrix<dtype::real>* mResiduum;
    Matrix<dtype::real>* mProjection;
    Matrix<dtype::real>* mRSOld;
    Matrix<dtype::real>* mRSNew;
    Matrix<dtype::real>* mTempVector;
    Matrix<dtype::real>* mTempNumber;
};

#endif
