// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create conjugate solver
Conjugate::Conjugate(dtype::size rows, cublasHandle_t handle, cudaStream_t stream)
    : mRows(rows), mResiduum(NULL), mProjection(NULL), mRSOld(NULL), mRSNew(NULL),
        mTempVector(NULL), mTempNumber(NULL) {
    // check input
    if (rows <= 1) {
        throw invalid_argument("Conjugate::Conjugate: rows <= 1");
    }
    if (handle == NULL) {
        throw invalid_argument("Conjugate::Conjugate: handle == NULL");
    }

    // create matrices
    this->mResiduum = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->mProjection = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->mRSOld = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->mRSNew = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->mTempVector = new Matrix<dtype::real>(this->rows(), this->mResiduum->dataRows() /
        Matrix<dtype::real>::blockSize, stream);
    this->mTempNumber = new Matrix<dtype::real>(this->rows(), 1, stream);
}

// release solver
Conjugate::~Conjugate() {
    // release matrices
    delete this->mResiduum;
    delete this->mProjection;
    delete this->mRSOld;
    delete this->mRSNew;
    delete this->mTempVector;
    delete this->mTempNumber;
}

// solve conjugate
void Conjugate::solve(Matrix<dtype::real>* A, Matrix<dtype::real>* x, Matrix<dtype::real>* f,
    dtype::size iterations, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (A == NULL) {
        throw invalid_argument("Conjugate::solve: A == NULL");
    }
    if (x == NULL) {
        throw invalid_argument("Conjugate::solve: x == NULL");
    }
    if (f == NULL) {
        throw invalid_argument("Conjugate::solve: f == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Conjugate::solve: handle == NULL");
    }

    // temp for pointer swap
    Matrix<dtype::real>* temp = NULL;

    // calc residuum r = f - A * x
    this->mResiduum->multiply(A, x, handle, stream);
    this->mResiduum->scalarMultiply(-1.0, stream);
    this->mResiduum->add(f, stream);

    // p = r
    this->mProjection->copy(this->mResiduum, stream);

    // calc rsold
    this->mRSOld->vectorDotProduct(this->mResiduum, this->mResiduum, stream);

    // iterate
    for (dtype::size i = 0; i < iterations; i++) {
        // calc A * p
        Conjugate::gemv(this->mTempVector, A, this->mProjection, stream);

        // calc p * A * p
        this->mTempNumber->vectorDotProduct(this->mProjection, this->mTempVector, stream);

        // update residuum
        Conjugate::updateVector(this->mResiduum, this->mResiduum, -1.0f,
            this->mTempVector, this->mRSOld, this->mTempNumber, stream);

        // update x
        Conjugate::updateVector(x, x, 1.0f, this->mProjection, this->mRSOld,
            this->mTempNumber, stream);

        // calc rsnew
        this->mRSNew->vectorDotProduct(this->mResiduum, this->mResiduum, stream);

        // update projection
        Conjugate::updateVector(this->mProjection, this->mResiduum, 1.0f,
            this->mProjection, this->mRSNew, this->mRSOld, stream);

        // swap rsold and rsnew
        temp = this->mRSOld;
        this->mRSOld = this->mRSNew;
        this->mRSNew = temp;
    }
}

