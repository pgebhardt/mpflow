// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "mpflow/mpflow.h"

// create conjugate solver
mpFlow::numeric::SparseConjugate::SparseConjugate(dtype::size rows,
    dtype::size columns, cudaStream_t stream)
    : rows_(rows), columns_(columns) {
    // check input
    if (rows <= 1) {
        throw std::invalid_argument("mpFlow::numeric::SparseConjugate::SparseConjugate: rows <= 1");
    }
    if (columns <= 1) {
        throw std::invalid_argument("mpFlow::numeric::SparseConjugate::SparseConjugate: columns <= 1");
    }

    // create matrices
    this->residuum_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->projection_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->rsold_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->rsnew_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->temp_vector_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->temp_number_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
}

// solve conjugate sparse
void mpFlow::numeric::SparseConjugate::solve(const std::shared_ptr<SparseMatrix<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations, bool dcFree,
    cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::SparseConjugate::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::SparseConjugate::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::SparseConjugate::solve: x == nullptr");
    }

    // calc residuum r = f - A * x
    A->multiply(x, stream, this->residuum());

    // regularize for dc free solution
    if (dcFree == true) {
        this->temp_number()->sum(x, stream);
        conjugate::addScalar(this->temp_number(), this->rows(), this->columns(),
            stream, this->residuum());
    }

    this->residuum()->scalarMultiply(-1.0, stream);
    this->residuum()->add(f, stream);

    // p = r
    this->projection()->copy(this->residuum(), stream);

    // calc rsold
    this->rsold()->vectorDotProduct(this->residuum(), this->residuum(), stream);

    // iterate
    for (dtype::index step = 0; step < iterations; ++step) {
        // calc A * p
        A->multiply(this->projection(), stream, this->temp_vector());

        // regularize for dc free solution
        if (dcFree == true) {
            this->temp_number()->sum(this->projection(), stream);
            conjugate::addScalar(this->temp_number(), this->rows(), this->columns(),
                stream, this->temp_vector());
        }

        // calc p * A * p
        this->temp_number()->vectorDotProduct(this->projection(),
            this->temp_vector(), stream);

        // update residuum
        conjugate::updateVector(this->residuum(), -1.0f, this->temp_vector(),
            this->rsold(), this->temp_number(), stream, this->residuum());

        // update x
        conjugate::updateVector(x, 1.0f, this->projection(), this->rsold(),
            this->temp_number(), stream, x);

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        conjugate::updateVector(this->residuum(), 1.0f, this->projection(),
            this->rsnew(), this->rsold(), stream, this->projection());

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew(), stream);
    }
}
