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
void SparseConjugate::solve(SparseMatrix& A, Matrix<dtype::real>& x, Matrix<dtype::real>& f,
    dtype::size iterations, bool dcFree, cudaStream_t stream) {
    // temp for pointer swap
    Matrix<dtype::real>* temp = NULL;

    // calc mResiduum r = f - A * x
    A.multiply(*this->mResiduum, x, stream);

    // regularize for dc free solution
    if (dcFree == true) {
        this->mTempNumber->sum(x, stream);
        Conjugate::addScalar(*this->mResiduum, *this->mTempNumber, this->rows(),
            this->columns(), stream);
    }

    this->mResiduum->scalarMultiply(-1.0, stream);
    this->mResiduum->add(f, stream);

    // p = r
    this->mProjection->add(*this->mResiduum, stream);

    // calc mRSOld
    this->mRSOld->vectorDotProduct(*this->mResiduum, *this->mResiduum, stream);

    // iterate
    for (dtype::size i = 0; i < iterations; i++) {
        // calc A * p
        A.multiply(*this->mTempVector, *this->mProjection, stream);

        // regularize for dc free solution
        if (dcFree == true) {
            this->mTempNumber->sum(*this->mProjection, stream);
            Conjugate::addScalar(*this->mTempVector, *this->mTempNumber, this->rows(),
                this->columns(), stream);
        }

        // calc p * A * p
        this->mTempNumber->vectorDotProduct(*this->mProjection, *this->mTempVector, stream);

        // update mResiduum
        Conjugate::updateVector(*this->mResiduum, *this->mResiduum,
            -1.0f, *this->mTempVector, *this->mRSOld, *this->mTempNumber, stream);

        // update x
        Conjugate::updateVector(x, x, 1.0f, *this->mProjection,
            *this->mRSOld, *this->mTempNumber, stream);

        // calc mRSNew
        this->mRSNew->vectorDotProduct(*this->mResiduum, *this->mResiduum, stream);

        // update mProjection
        Conjugate::updateVector(*this->mProjection, *this->mResiduum,
            1.0f, *this->mProjection, *this->mRSNew, *this->mRSOld, stream);

        // swap mRSOld and mRSNew
        temp = this->mRSOld;
        this->mRSOld = this->mRSNew;
        this->mRSNew = temp;
    }
}
