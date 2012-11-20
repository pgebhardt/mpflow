// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create conjugate solver
Conjugate::Conjugate(linalgcuSize_t rows, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (rows <= 1) {
        throw invalid_argument("Conjugate::Conjugate: rows <= 1");
    }
    if (handle == NULL) {
        throw invalid_argument("Conjugate::Conjugate: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init struct
    this->mRows = rows;
    this->mResiduum = NULL;
    this->mProjection = NULL;
    this->mRSOld = NULL;
    this->mRSNew = NULL;
    this->mTempVector = NULL;
    this->mTempNumber = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&this->mResiduum, this->rows(), 1, stream);
    error |= linalgcu_matrix_create(&this->mProjection, this->rows(), 1, stream);
    error |= linalgcu_matrix_create(&this->mRSOld, this->rows(), 1, stream);
    error |= linalgcu_matrix_create(&this->mRSNew, this->rows(), 1, stream);
    error |= linalgcu_matrix_create(&this->mTempVector, this->rows(), this->mResiduum->rows /
        LINALGCU_BLOCK_SIZE, stream);
    error |= linalgcu_matrix_create(&this->mTempNumber, this->rows(), 1, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Conjugate::Conjugate: create matrices");
    }
}

// release solver
Conjugate::~Conjugate() {
    // release matrices
    linalgcu_matrix_release(&this->mResiduum);
    linalgcu_matrix_release(&this->mProjection);
    linalgcu_matrix_release(&this->mRSOld);
    linalgcu_matrix_release(&this->mRSNew);
    linalgcu_matrix_release(&this->mTempVector);
    linalgcu_matrix_release(&this->mTempNumber);
}

// solve conjugate
void Conjugate::solve(linalgcuMatrix_t A, linalgcuMatrix_t x, linalgcuMatrix_t f,
    linalgcuSize_t iterations, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (A == NULL) {
        throw invalid_argument("Conjugate::solve: x == NULL");
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

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // temp for pointer swap
    linalgcuMatrix_t temp = NULL;

    // calc residuum r = f - A * x
    error  = linalgcu_matrix_multiply(this->mResiduum, A, x, handle, stream);
    error |= linalgcu_matrix_scalar_multiply(this->mResiduum, -1.0, stream);
    error |= linalgcu_matrix_add(this->mResiduum, f, stream);

    // p = r
    error |= linalgcu_matrix_copy(this->mProjection, this->mResiduum, stream);

    // calc rsold
    error |= linalgcu_matrix_vector_dot_product(this->mRSOld, this->mResiduum,
        this->mResiduum, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Conjugate::solve: setup solver");
    }

    // iterate
    for (linalgcuSize_t i = 0; i < iterations; i++) {
        // calc A * p
        Conjugate::gemv(this->mTempVector, A, this->mProjection, stream);

        // calc p * A * p
        error  = linalgcu_matrix_vector_dot_product(this->mTempNumber, this->mProjection,
            this->mTempVector, stream);

        // update residuum
        Conjugate::update_vector(this->mResiduum, this->mResiduum, -1.0f,
            this->mTempVector, this->mRSOld, this->mTempNumber, stream);

        // update x
        Conjugate::update_vector(x, x, 1.0f, this->mProjection, this->mRSOld,
            this->mTempNumber, stream);

        // calc rsnew
        error |= linalgcu_matrix_vector_dot_product(this->mRSNew, this->mResiduum,
            this->mResiduum, stream);

        // update projection
        Conjugate::update_vector(this->mProjection, this->mResiduum, 1.0f,
            this->mProjection, this->mRSNew, this->mRSOld, stream);

        // swap rsold and rsnew
        temp = this->mRSOld;
        this->mRSOld = this->mRSNew;
        this->mRSNew = temp;

        // check success
        if (error != LINALGCU_SUCCESS) {
            throw logic_error("Conjugate::solve: iterate");
        }
    }
}

