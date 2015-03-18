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
    class dataType,
    template <class type> class matrixType
>
mpFlow::numeric::ConjugateGradient<dataType, matrixType>::ConjugateGradient(
    unsigned const rows, unsigned const cols, cudaStream_t const stream)
    : rows(rows), cols(cols) {
    // check input
    if (rows < 1) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::ConjugateGradient: rows <= 1");
    }
    if (cols < 1) {
        throw std::invalid_argument("mpFlow::numeric::ConjugateGradient::ConjugateGradient: cols <= 1");
    }

    // create matrices
    this->r = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->p = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->roh = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0);
    this->rohOld = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->temp1 = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->temp2 = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
}

// solve conjugateGradient spars conste
template <
    class dataType,
    template <class type> class matrixType
>
unsigned mpFlow::numeric::ConjugateGradient<dataType, matrixType>::solve(
    std::shared_ptr<matrixType<dataType> const> const A,
    std::shared_ptr<Matrix<dataType> const> const f, unsigned const iterations,
    cublasHandle_t const handle, cudaStream_t const stream, std::shared_ptr<Matrix<dataType>> const x,
    double const tolerance, bool const dcFree) {
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
        addScalar(this->temp1, this->rows, this->cols, stream, this->r);
    }

    this->r->scalarMultiply(-1.0, stream);
    this->r->add(f, stream);

    // p = r
    this->p->copy(this->r, stream);

    // calc rsold
    this->rohOld->vectorDotProduct(this->r, this->r, stream);

    // iterate
    for (unsigned step = 0; step < iterations; ++step) {
        // calc A * p
        this->temp1->multiply(A, this->p, handle, stream);

        // regularize for dc free solution
        if (dcFree == true) {
            this->temp2->sum(this->p, stream);
            addScalar(this->temp2, this->rows, this->cols, stream, this->temp1);
        }

        // calc p * A * p
        this->temp2->vectorDotProduct(this->p, this->temp1, stream);

        // update residuum
        updateVector(this->r, -1.0, this->temp1, this->rohOld, this->temp2, stream, this->r);

        // update x
        updateVector(x, 1.0, this->p, this->rohOld, this->temp2, stream, x);

        // calc rsnew
        this->roh->vectorDotProduct(this->r, this->r, stream);

        // check error bound for all column vectors of residuum
        if (tolerance > 0.0) {
            using namespace std;
            using namespace thrust;

            this->roh->copyToHost(stream);
            cudaStreamSynchronize(stream);

            for (unsigned i = 0; i < this->roh->cols; ++i) {
                if (abs(sqrt((*this->roh)(0, i))) >= tolerance) {
                    break;
                }
                return step + 1;
            }
        }

        // update projection
        updateVector(this->r, 1.0, this->p, this->roh, this->rohOld, stream, this->p);

        // copy rsnew to rsold
        this->rohOld->copy(this->roh, stream);
    }

    return iterations;
}

// add scalar
template <
    class dataType,
    template <class type> class matrixType
>
void mpFlow::numeric::ConjugateGradient<dataType, matrixType>::addScalar(
    std::shared_ptr<Matrix<dataType> const> const scalar,
    unsigned const rows, unsigned const columns, cudaStream_t const stream,
    std::shared_ptr<Matrix<dataType>> const vector) {
    // check input
    if (scalar == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: scalar == nullptr");
    }
    if (vector == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::conjugateGradient::addScalar: vector == nullptr");
    }

    // kernel dimension
    dim3 blocks(vector->dataRows / matrix::blockSize,
        vector->dataCols == 1 ? 1 : vector->dataCols / matrix::blockSize);
    dim3 threads(matrix::blockSize,
        vector->dataCols == 1 ? 1 : matrix::blockSize);

    // execute kernel
    conjugateGradientKernel::addScalar<dataType>(blocks, threads, stream, scalar->deviceData,
        vector->dataRows, rows, columns, vector->deviceData);
}

// update vector
template <
    class dataType,
    template <class type> class matrixType
>
void mpFlow::numeric::ConjugateGradient<dataType, matrixType>::updateVector(
    std::shared_ptr<Matrix<dataType> const> const x1, double const sign,
    std::shared_ptr<Matrix<dataType> const> const x2,
    std::shared_ptr<Matrix<dataType> const> const r1,
    std::shared_ptr<Matrix<dataType> const> const r2, cudaStream_t const stream,
    std::shared_ptr<Matrix<dataType>> const result) {
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
    dim3 blocks(result->dataRows / matrix::blockSize,
        result->dataCols == 1 ? 1 : result->dataCols / matrix::blockSize);
    dim3 threads(matrix::blockSize, result->dataCols == 1 ? 1 : matrix::blockSize);

    // execute kernel
    conjugateGradientKernel::updateVector<dataType>(blocks, threads, stream, x1->deviceData, sign,
        x2->deviceData, r1->deviceData, r2->deviceData, result->dataRows,
        result->deviceData);
}

// specialisations
template class mpFlow::numeric::ConjugateGradient<float, mpFlow::numeric::Matrix>;
template class mpFlow::numeric::ConjugateGradient<float, mpFlow::numeric::SparseMatrix>;
template class mpFlow::numeric::ConjugateGradient<double, mpFlow::numeric::Matrix>;
template class mpFlow::numeric::ConjugateGradient<double, mpFlow::numeric::SparseMatrix>;
template class mpFlow::numeric::ConjugateGradient<thrust::complex<float>, mpFlow::numeric::Matrix>;
template class mpFlow::numeric::ConjugateGradient<thrust::complex<float>, mpFlow::numeric::SparseMatrix>;
template class mpFlow::numeric::ConjugateGradient<thrust::complex<double>, mpFlow::numeric::Matrix>;
template class mpFlow::numeric::ConjugateGradient<thrust::complex<double>, mpFlow::numeric::SparseMatrix>;
