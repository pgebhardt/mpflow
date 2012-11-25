// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create conjugate solver
SparseConjugate::SparseConjugate(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : mRows(rows), mColumns(columns), mResiduum(NULL), mProjection(NULL), mRSOld(NULL),
        mRSNew(NULL), mTempVector(NULL), mTempNumber(NULL) {
    // check input
    if (rows <= 1) {
        throw invalid_argument("SparseConjugate::SparseConjugate: rows <= 1");
    }
    if (columns <= 1) {
        throw invalid_argument("SparseConjugate::SparseConjugate: columns <= 1");
    }

    // create matrices
    this->mResiduum = new Matrix<dtype::real>(this->rows(), this->columns(), stream);
    this->mProjection = new Matrix<dtype::real>(this->rows(), this->columns(), stream);
    this->mRSOld = new Matrix<dtype::real>(this->rows(), this->columns(), stream);
    this->mRSNew = new Matrix<dtype::real>(this->rows(), this->columns(), stream);
    this->mTempVector = new Matrix<dtype::real>(this->rows(), this->columns(), stream);
    this->mTempNumber = new Matrix<dtype::real>(this->rows(), this->columns(), stream);
}

// release solver
SparseConjugate::~SparseConjugate() {
    // release matrices
    delete this->mResiduum;
    delete this->mProjection;
    delete this->mRSOld;
    delete this->mRSNew;
    delete this->mTempVector;
    delete this->mTempNumber;
}

// solve conjugate sparse
void SparseConjugate::solve(SparseMatrix* A, Matrix<dtype::real>* x, Matrix<dtype::real>* f,
    dtype::size iterations, bool dcFree, cudaStream_t stream) {
    // check input
    if (A == NULL) {
        throw invalid_argument("SparseConjugate::solve: A == NULL");
    }
    if (x == NULL) {
        throw invalid_argument("SparseConjugate::solve: x == NULL");
    }
    if (f == NULL) {
        throw invalid_argument("SparseConjugate::solve: f == NULL");
    }

    // calc residuum r = f - A * x
    A->multiply(this->residuum(), x, stream);

    // regularize for dc free solution
    if (dcFree == true) {
        this->tempNumber()->sum(x, stream);
        Conjugate::addScalar(this->residuum(), this->tempNumber(), this->rows(),
            this->columns(), stream);
    }

    this->residuum()->scalarMultiply(-1.0, stream);
    this->residuum()->add(f, stream);

    // p = r
    this->projection()->copy(this->residuum(), stream);

    // calc rsold
    this->rsold()->vectorDotProduct(this->residuum(), this->residuum(), stream);

    // iterate
    for (dtype::size i = 0; i < iterations; i++) {
        // calc A * p
        A->multiply(this->tempVector(), this->projection(), stream);

        // regularize for dc free solution
        if (dcFree == true) {
            this->tempNumber()->sum(this->projection(), stream);
            Conjugate::addScalar(this->tempVector(), this->tempNumber(), this->rows(),
                this->columns(), stream);
        }

        // calc p * A * p
        this->tempNumber()->vectorDotProduct(this->projection(), this->tempVector(), stream);

        // update residuum
        Conjugate::updateVector(this->residuum(), this->residuum(),
            -1.0f, this->tempVector(), this->rsold(), this->tempNumber(), stream);

        // update x
        Conjugate::updateVector(x, x, 1.0f, this->projection(),
            this->rsold(), this->tempNumber(), stream);

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        Conjugate::updateVector(this->projection(), this->residuum(),
            1.0f, this->projection(), this->rsnew(), this->rsold(), stream);

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew());
    }
}
