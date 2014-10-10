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
    template <class type> class matrixType
>
mpFlow::numeric::ConjugateGradient<matrixType>::ConjugateGradient(dtype::size rows, dtype::size cols, cudaStream_t stream)
    : rows(rows), cols(cols) {
    // check input
    if (rows < 1) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::ConjugateGradient: rows <= 1");
    }
    if (cols < 1) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::ConjugateGradient: cols <= 1");
    }

    // create matrices
    this->r = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 0.0, false);
    this->p = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 0.0, false);
    this->roh = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->rohOld = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 0.0, false);
    this->temp1 = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 0.0, false);
    this->temp2 = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 0.0, false);
}

// solve conjugateGradient sparse
template <
    template <class type> class matrixType
>
void mpFlow::numeric::ConjugateGradient<matrixType>::solve(
    const std::shared_ptr<matrixType<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations,
    cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x,
    dtype::real tolerance, bool dcFree) {
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
    this->r->multiply(A, x, handle, stream);

    // regularize for dc free solution
    if (dcFree == true) {
        this->temp1->sum(x, stream);
        conjugateGradient::addScalar(this->temp1, this->rows, this->cols,
            stream, this->r);
    }

    this->r->scalarMultiply(-1.0, stream);
    this->r->add(f, stream);

    // p = r
    this->p->copy(this->r, stream);

    // calc rsold
    this->rohOld->vectorDotProduct(this->r, this->r, stream);

    // iterate
    for (dtype::index step = 0; step < iterations; ++step) {
        // calc A * p
        this->temp1->multiply(A, this->p, handle, stream);

        // regularize for dc free solution
        if (dcFree == true) {
            this->temp2->sum(this->p, stream);
            conjugateGradient::addScalar(this->temp2, this->rows, this->cols,
                stream, this->temp1);
        }

        // calc p * A * p
        this->temp2->vectorDotProduct(this->p, this->temp1, stream);

        // update residuum
        conjugateGradient::updateVector(this->r, -1.0f, this->temp1,
            this->rohOld, this->temp2, stream, this->r);

        // update x
        conjugateGradient::updateVector(x, 1.0f, this->p, this->rohOld,
            this->temp2, stream, x);

        // calc rsnew
        this->roh->vectorDotProduct(this->r, this->r, stream);

        // check error bound for all column vectors of residuum
        if (tolerance > 0.0) {
            this->roh->copyToHost(stream);
            cudaStreamSynchronize(stream);

            for (dtype::index i = 0; i < this->roh->cols; ++i) {
                if (sqrt((*this->roh)(0, i)) >= 1e-6) {
                    break;
                }
                return;
            }
        }

        // update projection
        conjugateGradient::updateVector(this->r, 1.0f, this->p,
            this->roh, this->rohOld, stream, this->p);

        // copy rsnew to rsold
        this->rohOld->copy(this->roh, stream);
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
    dim3 blocks(vector->dataRows / matrix::block_size,
        vector->dataCols == 1 ? 1 : vector->dataCols / matrix::block_size);
    dim3 threads(matrix::block_size,
        vector->dataCols == 1 ? 1 : matrix::block_size);

    // execute kernel
    conjugateGradientKernel::addScalar(blocks, threads, stream, scalar->deviceData,
        vector->dataRows, rows, columns, vector->deviceData);
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
    dim3 blocks(result->dataRows / matrix::block_size,
        result->dataCols == 1 ? 1 : result->dataCols / matrix::block_size);
    dim3 threads(matrix::block_size, result->dataCols == 1 ? 1 : matrix::block_size);

    // execute kernel
    conjugateGradientKernel::updateVector(blocks, threads, stream, x1->deviceData, sign,
        x2->deviceData, r1->deviceData, r2->deviceData, result->dataRows,
        result->deviceData);
}

// specialisations
template class mpFlow::numeric::ConjugateGradient<mpFlow::numeric::Matrix>;
template class mpFlow::numeric::ConjugateGradient<mpFlow::numeric::SparseMatrix>;
