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
    this->mTempVector = new Matrix<dtype::real>(this->rows(), this->residuum()->dataRows() /
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

    // calc residuum r = f - A * x
    this->residuum()->multiply(A, x, handle, stream);
    this->residuum()->scalarMultiply(-1.0, stream);
    this->residuum()->add(f, stream);

    // p = r
    this->projection()->copy(this->residuum(), stream);

    // calc rsold
    this->rsold()->vectorDotProduct(this->residuum(), this->residuum(), stream);

    // iterate
    for (dtype::size i = 0; i < iterations; i++) {
        // calc A * p
        Conjugate::gemv(this->tempVector(), A, this->projection(), stream);

        // calc p * A * p
        this->tempNumber()->vectorDotProduct(this->projection(), this->tempVector(), stream);

        // update residuum
        Conjugate::updateVector(this->residuum(), this->residuum(), -1.0f,
            this->tempVector(), this->rsold(), this->tempNumber(), stream);

        // update x
        Conjugate::updateVector(x, x, 1.0f, this->projection(), this->rsold(),
            this->tempNumber(), stream);

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        Conjugate::updateVector(this->projection(), this->residuum(), 1.0f,
            this->projection(), this->rsnew(), this->rsold(), stream);

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew(), stream);
    }
}

