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
mpFlow::numeric::FastConjugate::FastConjugate(dtype::size rows, dtype::size, cudaStream_t stream)
    : rows_(rows)  {
    // check input
    if (rows <= 1) {
        throw std::invalid_argument("mpFlow::numeric::FastConjugate::FastConjugate: rows <= 1");
    }

    // create matrices
    this->residuum_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->projection_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->rsold_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->rsnew_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
    this->temp_vector_ = std::make_shared<Matrix<dtype::real>>(this->rows(),
        this->residuum()->data_rows() / matrix::block_size, stream);
    this->temp_number_ = std::make_shared<Matrix<dtype::real>>(this->rows(), 1, stream);
}

// solve conjugate
void mpFlow::numeric::FastConjugate::solve(const std::shared_ptr<Matrix<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations,
    cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::FastConjugate::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::FastConjugate::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::FastConjugate::solve: x == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("mpFlow::numeric::FastConjugate::solve: handle == NULL");
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

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        conjugate::updateVector(this->residuum(), 1.0f, this->projection(),
            this->rsnew(), this->rsold(), stream, this->projection());

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew(), stream);
    }
}
