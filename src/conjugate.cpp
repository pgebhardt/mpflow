// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"
#include "../include/conjugate.hcu"
#include "../include/conjugate.hpp"

// create conjugate solver
fastEIT::numeric::Conjugate::Conjugate(dtype::size rows, cublasHandle_t handle,
    cudaStream_t stream)
    : rows_(rows), residuum_(NULL), projection_(NULL), rsold_(NULL), rsnew_(NULL),
        temp_vector_(NULL), temp_number_(NULL) {
    // check input
    if (rows <= 1) {
        throw std::invalid_argument("Conjugate::Conjugate: rows <= 1");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Conjugate::Conjugate: handle == NULL");
    }

    // create matrices
    this->residuum_ = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->projection_ = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->rsold_ = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->rsnew_ = new Matrix<dtype::real>(this->rows(), 1, stream);
    this->temp_vector_ = new Matrix<dtype::real>(this->rows(), this->residuum().data_rows() /
        Matrix<dtype::real>::block_size, stream);
    this->temp_number_ = new Matrix<dtype::real>(this->rows(), 1, stream);
}

// release solver
fastEIT::numeric::Conjugate::~Conjugate() {
    // release matrices
    delete this->residuum_;
    delete this->projection_;
    delete this->rsold_;
    delete this->rsnew_;
    delete this->temp_vector_;
    delete this->temp_number_;
}

// solve conjugate
void fastEIT::numeric::Conjugate::solve(const Matrix<dtype::real>& A,
    const Matrix<dtype::real>& f, dtype::size iterations, cublasHandle_t handle,
    cudaStream_t stream, Matrix<dtype::real>* x) {
    // check input
    if (x == NULL) {
        throw std::invalid_argument("Conjugate::solve: x == NULL");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Conjugate::solve: handle == NULL");
    }

    // calc residuum r = f - A * x
    this->residuum().multiply(A, *x, handle, stream);
    this->residuum().scalarMultiply(-1.0, stream);
    this->residuum().add(f, stream);

    // p = r
    this->projection().copy(this->residuum(), stream);

    // calc rsold
    this->rsold().vectorDotProduct(this->residuum(), this->residuum(), stream);

    // iterate
    for (dtype::index step = 0; step < iterations; ++step) {
        // calc A * p
        conjugate::gemv(A, this->projection(), stream, &this->temp_vector());

        // calc p * A * p
        this->temp_number().vectorDotProduct(this->projection(), this->temp_vector(), stream);

        // update residuum
        conjugate::updateVector(this->residuum(), -1.0f, this->temp_vector(),
            this->rsold(), this->temp_number(), stream, &this->residuum());

        // update x
        conjugate::updateVector(*x, 1.0f, this->projection(), this->rsold(),
            this->temp_number(), stream, x);

        // calc rsnew
        this->rsnew().vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        conjugate::updateVector(this->residuum(), 1.0f, this->projection(),
            this->rsnew(), this->rsold(), stream, &this->projection());

        // copy rsnew to rsold
        this->rsold().copy(this->rsnew(), stream);
    }
}

