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
#include "mpflow/numeric/bicgstab_kernel.h"

// create bicgstab solver
template <
    template <class type> class matrixType
>
mpFlow::numeric::BiCGSTAB<matrixType>::BiCGSTAB(dtype::size rows, dtype::size cols, cudaStream_t stream)
    : rows(rows), cols(cols) {
    // check input
    if (rows < 1) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::BiCGSTAB: rows <= 1");
    }
    if (cols < 1) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::BiCGSTAB: cols <= 1");
    }

    // create matrices
    this->r = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->rHat = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 1.0);
    this->roh = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 1.0);
    this->rohOld = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 1.0);
    this->alpha = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 1.0);
    this->beta = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->omega = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream, 1.0);
    this->nu = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->p = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->t = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->s = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->error = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->temp1 = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
    this->temp2 = std::make_shared<Matrix<dtype::real>>(this->rows, this->cols, stream);
}

// solve bicgstab sparse
template <
    template <class type> class matrixType
>
void mpFlow::numeric::BiCGSTAB<matrixType>::solve(
    const std::shared_ptr<matrixType<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations,
    cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x,
    dtype::real tolerance, bool) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::solve: x == nullptr");
    }

    // r0 = f - A * x0
    this->r->multiply(A, x, handle, stream);
    this->r->scalarMultiply(-1.0, stream);
    this->r->add(f, stream);

    // Choose an arbitrary vector rHat such that (r, rHat) != 0, e.g. rHat = r
    // this->rHat->copy(r, stream);

    // initialize current error vector
    this->error->vectorDotProduct(this->r, this->r, stream);
    this->error->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // iterate
    for (dtype::index step = 0; step < iterations; ++step) {
        // roh = (rHat, r)
        this->rohOld->copy(this->roh, stream);
        this->roh->vectorDotProduct(this->rHat, this->r, stream);

        // beta = (roh(i) / roh(i-1)) * (alpha / omega)
        this->temp1->elementwiseDivision(this->roh, this->rohOld, stream);
        this->temp2->elementwiseDivision(this->alpha, this->omega, stream);
        this->beta->elementwiseMultiply(this->temp1, this->temp2, stream);

        // p = r + beta * (p - omega * nu)
        bicgstab::updateVector(this->p, -1.0, this->nu, this->omega, stream,
            this->temp1);
        bicgstab::updateVector(this->r, 1.0, this->temp1, this->beta, stream,
            this->p);

        // nu = A * p
        this->nu->multiply(A, this->p, handle, stream);

        // alpha = roh / (rHat, nu)
        this->temp1->vectorDotProduct(this->rHat, this->nu, stream);
        this->alpha->elementwiseDivision(this->roh, this->temp1, stream);

        // s = r - alpha * nu
        bicgstab::updateVector(this->r, -1.0, this->nu, this->alpha, stream,
            this->s);

        // t = A * s
        this->t->multiply(A, this->s, handle, stream);

        // omega = (t, s) / (t, t)
        this->temp1->vectorDotProduct(this->t, this->s, stream);
        this->temp2->vectorDotProduct(this->t, this->t, stream);
        this->omega->elementwiseDivision(this->temp1, this->temp2, stream);

        // x = x + alpha * p + omega * s
        bicgstab::updateVector(x, 1.0, this->p, this->alpha, stream, x);
        bicgstab::updateVector(x, 1.0, this->s, this->omega, stream, x);

        // r = s - omega * t
        bicgstab::updateVector(this->s, -1.0, this->t, this->omega, stream,
            this->r);

        // check error bound for all column vectors of residuum
        if (tolerance > 0.0) {
            this->error->vectorDotProduct(this->r, this->r, stream);
            this->error->copyToHost(stream);
            cudaStreamSynchronize(stream);

            for (dtype::index i = 0; i < error->cols; ++i) {
                if (sqrt((*this->error)(0, i)) >= tolerance) {
                    break;
                }
                return;
            }
        }
    }
}

// update vector
void mpFlow::numeric::bicgstab::updateVector(
    const std::shared_ptr<Matrix<dtype::real>> x1, dtype::real sign,
    const std::shared_ptr<Matrix<dtype::real>> x2,
    const std::shared_ptr<Matrix<dtype::real>> scalar, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> result) {
    // check input
    if (x1 == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: x1 == nullptr");
    }
    if (x2 == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: x2 == nullptr");
    }
    if (scalar == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: scalar == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: result == nullptr");
    }

    // kernel dimension
    dim3 blocks(result->dataRows / matrix::block_size,
        result->dataCols == 1 ? 1 : result->dataCols / matrix::block_size);
    dim3 threads(matrix::block_size, result->dataCols == 1 ? 1 : matrix::block_size);

    // execute kernel
    bicgstabKernel::updateVector(blocks, threads, stream, x1->deviceData, sign,
        x2->deviceData, scalar->deviceData, result->dataRows, result->deviceData);
}

// specialisations
template class mpFlow::numeric::BiCGSTAB<mpFlow::numeric::Matrix>;
template class mpFlow::numeric::BiCGSTAB<mpFlow::numeric::SparseMatrix>;
