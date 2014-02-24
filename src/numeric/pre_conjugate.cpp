// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"

// create conjugate solver
mpFlow::numeric::PreConjugate::PreConjugate(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : rows_(rows), columns_(columns) {
    // check input
    if (rows < 1) {
        throw std::invalid_argument("mpFlow::numeric::PreConjugate::PreConjugate: rows <= 1");
    }
    if (columns < 1) {
        throw std::invalid_argument("mpFlow::numeric::PreConjugate::PreConjugate: columns <= 1");
    }

    // create matrices
    this->residuum_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->projection_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->z_ = std::make_shared<Matrix<dtype::real>>(this->rows(),
        this->residuum()->data_rows() / matrix::block_size, stream);
    this->rsold_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->rsnew_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->temp_vector_ = std::make_shared<Matrix<dtype::real>>(this->rows(),
        this->residuum()->data_rows() / matrix::block_size, stream);
    this->temp_number_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->preconditioner_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->rows(),
        stream);

    // init preconditioner as unit matrix
    for (dtype::index row = 0; row < this->rows(); ++row) {
        (*this->preconditioner())(row, row) = 1.0;
    }
    this->preconditioner()->copyToDevice(stream);
}

// solve conjugate sparse
void mpFlow::numeric::PreConjugate::solve(const std::shared_ptr<Matrix<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations,
    cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::PreConjugate::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::PreConjugate::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::PreConjugate::solve: x == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::PreConjugate::solve: handle == nullptr");
    }

    // calc residuum r = f - A * x
    this->residuum()->multiply(A, x, handle, stream);
    this->residuum()->scalarMultiply(-1.0, stream);
    this->residuum()->add(f, stream);

    // apply preconditioner
    conjugate::gemv(this->preconditioner(), this->residuum(), stream, this->z());

    // p = z
    // this->projection()->copy(this->z(), stream);
    this->projection()->multiply(this->preconditioner(), this->residuum(), handle, stream);

    // calc rsold
    this->rsold()->vectorDotProduct(this->residuum(), this->z(), stream);

    // iterate
    for (dtype::index step = 0; step < iterations; ++step) {
        // calc A * p
        conjugate::gemv(A, this->projection(), stream, this->temp_vector());

        // calc p * A * p
        this->temp_number()->vectorDotProduct(this->projection(),
            this->temp_vector(), stream);

        // update residuum
        conjugate::updateVector(this->residuum(), -1.0f, this->temp_vector(),
            this->rsold(), this->temp_number(), stream, this->residuum());

        // update x
        conjugate::updateVector(x, 1.0f, this->projection(), this->rsold(),
            this->temp_number(), stream, x);

        // apply preconditioner
        conjugate::gemv(this->preconditioner(), this->residuum(), stream, this->z());

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->z(), this->residuum(), stream);

        // update projection
        conjugate::updateVector(this->z(), 1.0f, this->projection(),
            this->rsnew(), this->rsold(), stream, this->projection());

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew(), stream);
    }
}
