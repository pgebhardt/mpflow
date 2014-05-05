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
#include "mpflow/numeric/conjugate_gradient_kernel.h"

// create conjugateGradient solver
template <
    template <class type> class matrix_type
>
mpFlow::numeric::ConjugateGradient<matrix_type>::ConjugateGradient(dtype::size rows, dtype::size columns, cudaStream_t stream)
    : rows_(rows), columns_(columns) {
    // check input
    if (rows < 1) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::ConjugateGradient: rows <= 1");
    }
    if (columns < 1) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::ConjugateGradient: columns <= 1");
    }

    // create matrices
    this->residuum_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->projection_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->rsold_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->rsnew_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->temp_vector_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
    this->temp_number_ = std::make_shared<Matrix<dtype::real>>(this->rows(), this->columns(), stream);
}

// solve conjugateGradient sparse
template <
    template <class type> class matrix_type
>
void mpFlow::numeric::ConjugateGradient<matrix_type>::solve(const std::shared_ptr<matrix_type<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations, bool dcFree,
    cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::solve: x == nullptr");
    }

    // calc residuum r = f - A * x
    this->residuum()->multiply(A, x, handle, stream);

    // regularize for dc free solution
    if (dcFree == true) {
        this->temp_number()->sum(x, stream);
        conjugateGradient::addScalar(this->temp_number(), this->rows(), this->columns(),
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
        this->temp_vector()->multiply(A, this->projection(), handle, stream);

        // regularize for dc free solution
        if (dcFree == true) {
            this->temp_number()->sum(this->projection(), stream);
            conjugateGradient::addScalar(this->temp_number(), this->rows(), this->columns(),
                stream, this->temp_vector());
        }

        // calc p * A * p
        this->temp_number()->vectorDotProduct(this->projection(),
            this->temp_vector(), stream);

        // update residuum
        conjugateGradient::updateVector(this->residuum(), -1.0f, this->temp_vector(),
            this->rsold(), this->temp_number(), stream, this->residuum());

        // update x
        conjugateGradient::updateVector(x, 1.0f, this->projection(), this->rsold(),
            this->temp_number(), stream, x);

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        conjugateGradient::updateVector(this->residuum(), 1.0f, this->projection(),
            this->rsnew(), this->rsold(), stream, this->projection());

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew(), stream);
    }
}

// add scalar
void mpFlow::numeric::conjugateGradient::addScalar(
    const std::shared_ptr<Matrix<dtype::real>> scalar,
    dtype::size rows, dtype::size columns, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> vector) {
    // check input
    if (scalar == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: scalar == nullptr");
    }
    if (vector == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: vector == nullptr");
    }

    // kernel dimension
    dim3 blocks(vector->data_rows() / matrix::block_size,
        vector->data_columns() == 1 ? 1 :
        vector->data_columns() / matrix::block_size);
    dim3 threads(matrix::block_size,
        vector->data_columns() == 1 ? 1 : matrix::block_size);

    // execute kernel
    conjugateGradientKernel::addScalar(blocks, threads, stream, scalar->device_data(),
        vector->data_rows(), rows, columns, vector->device_data());
}

// update vector
void mpFlow::numeric::conjugateGradient::updateVector(
    const std::shared_ptr<Matrix<dtype::real>> x1, dtype::real sign,
    const std::shared_ptr<Matrix<dtype::real>> x2,
    const std::shared_ptr<Matrix<dtype::real>> r1,
    const std::shared_ptr<Matrix<dtype::real>> r2, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> result) {
    // check input
    if (x1 == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: x1 == nullptr");
    }
    if (x2 == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: x2 == nullptr");
    }
    if (r1 == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: r1 == nullptr");
    }
    if (r2 == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: r2 == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: result == nullptr");
    }

    // kernel dimension
    dim3 blocks(result->data_rows() / matrix::block_size,
        result->data_columns() == 1 ? 1 :
        result->data_columns() / matrix::block_size);
    dim3 threads(matrix::block_size,
        result->data_columns() == 1 ? 1 : matrix::block_size);

    // execute kernel
    conjugateGradientKernel::updateVector(blocks, threads, stream, x1->device_data(), sign,
        x2->device_data(), r1->device_data(), r2->device_data(), result->data_rows(),
        result->device_data());
}

// specialisations
template class mpFlow::numeric::ConjugateGradient<mpFlow::numeric::Matrix>;
template class mpFlow::numeric::ConjugateGradient<mpFlow::numeric::SparseMatrix>;
