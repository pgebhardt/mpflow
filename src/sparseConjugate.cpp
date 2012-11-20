// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create conjugate solver
SparseConjugate::SparseConjugate(linalgcuSize_t rows, linalgcuSize_t columns, cudaStream_t stream) {
    // check input
    if (rows <= 1) {
        throw invalid_argument("SparseConjugate::SparseConjugate: rows <= 1");
    }
    if (columns <= 1) {
        throw invalid_argument("SparseConjugate::SparseConjugate: columns <= 1");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init member
    this->mRows = rows;
    this->mColumns = columns;
    this->mResiduum = NULL;
    this->mProjection = NULL;
    this->mRSOld = NULL;
    this->mRSNew = NULL;
    this->mTempVector = NULL;
    this->mTempNumber = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&this->mResiduum, this->rows(), this->columns(), stream);
    error |= linalgcu_matrix_create(&this->mProjection, this->rows(), this->columns(), stream);
    error |= linalgcu_matrix_create(&this->mRSOld, this->rows(), this->columns(), stream);
    error |= linalgcu_matrix_create(&this->mRSNew, this->rows(), this->columns(), stream);
    error |= linalgcu_matrix_create(&this->mTempVector, this->rows(), this->columns(), stream);
    error |= linalgcu_matrix_create(&this->mTempNumber, this->rows(), this->columns(), stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("SparseConjugate::SparseConjugate: create matrices");
    }
}

// release solver
SparseConjugate::~SparseConjugate() {
    // release matrices
    linalgcu_matrix_release(&this->mResiduum);
    linalgcu_matrix_release(&this->mProjection);
    linalgcu_matrix_release(&this->mRSOld);
    linalgcu_matrix_release(&this->mRSNew);
    linalgcu_matrix_release(&this->mTempVector);
    linalgcu_matrix_release(&this->mTempNumber);
}

// solve conjugate sparse
void SparseConjugate::solve(linalgcuSparseMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, bool dcFree, cudaStream_t stream) {
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

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // temp for pointer swap
    linalgcuMatrix_t temp = NULL;

    // calc mResiduum r = f - A * x
    error  = linalgcu_sparse_matrix_multiply(this->mResiduum, A, x, stream);

    // regularize for dc free solution
    if (dcFree == true) {
        error |= linalgcu_matrix_sum(this->mTempNumber, x, stream);
        Conjugate::add_scalar(this->mResiduum, this->mTempNumber, this->rows(),
            this->columns(), stream);
    }

    error |= linalgcu_matrix_scalar_multiply(this->mResiduum, -1.0, stream);
    error |= linalgcu_matrix_add(this->mResiduum, f, stream);

    // p = r
    error |= linalgcu_matrix_copy(this->mProjection, this->mResiduum, stream);

    // calc mRSOld
    error |= linalgcu_matrix_vector_dot_product(this->mRSOld, this->mResiduum,
        this->mResiduum, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("SparseConjugate::solve: setup solver");
    }

    // iterate
    for (linalgcuSize_t i = 0; i < iterations; i++) {
        // calc A * p
        error  = linalgcu_sparse_matrix_multiply(this->mTempVector, A, this->mProjection,
            stream);

        // regularize for dc free solution
        if (dcFree == true) {
            error |= linalgcu_matrix_sum(this->mTempNumber, this->mProjection, stream);
            Conjugate::add_scalar(this->mTempVector, this->mTempNumber, this->rows(),
                this->columns(), stream);
        }

        // calc p * A * p
        error |= linalgcu_matrix_vector_dot_product(this->mTempNumber, this->mProjection,
            this->mTempVector, stream);

        // update mResiduum
        Conjugate::update_vector(this->mResiduum, this->mResiduum,
            -1.0f, this->mTempVector, this->mRSOld, this->mTempNumber, stream);

        // update x
        Conjugate::update_vector(x, x, 1.0f, this->mProjection,
            this->mRSOld, this->mTempNumber, stream);

        // calc mRSNew
        error |= linalgcu_matrix_vector_dot_product(this->mRSNew, this->mResiduum,
            this->mResiduum, stream);

        // update mProjection
        Conjugate::update_vector(this->mProjection, this->mResiduum,
            1.0f, this->mProjection, this->mRSNew, this->mRSOld, stream);

        // swap mRSOld and mRSNew
        temp = this->mRSOld;
        this->mRSOld = this->mRSNew;
        this->mRSNew = temp;

        // check success
        if (error != LINALGCU_SUCCESS) {
            throw logic_error("SparseConjugate::solve: iterate");
        }
    }
}
