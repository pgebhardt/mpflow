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
    class dataType
>
mpFlow::numeric::BiCGSTAB<dataType>::BiCGSTAB(
    unsigned const rows, unsigned const cols, cudaStream_t const stream)
    : rows(rows), cols(cols) {
    // check input
    if (rows < 1) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::BiCGSTAB: rows <= 1");
    }
    if (cols < 1) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::BiCGSTAB: cols <= 1");
    }

    // create matrices
    this->r = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->rHat = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->roh = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 1.0, false);
    this->rohOld = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 1.0, false);
    this->alpha = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 1.0, false);
    this->beta = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->omega = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 1.0, false);
    this->nu = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->p = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->t = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->s = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->y = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->z = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->error = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream);
    this->reference = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream);
    this->temp1 = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
    this->temp2 = std::make_shared<Matrix<dataType>>(this->rows, this->cols, stream, 0.0, false);
}

// solve bicgstab sparse
template <
    class dataType
>
template <
    template <class type> class matrixType
>
unsigned mpFlow::numeric::BiCGSTAB<dataType>::solve(std::shared_ptr<matrixType<dataType>> const A,
    std::shared_ptr<Matrix<dataType> const> const b, cublasHandle_t const handle,
    cudaStream_t const stream, std::shared_ptr<Matrix<dataType>> const x,
    std::shared_ptr<matrixType<dataType>> const K, bool const,
    unsigned const maxIterations, double const tolerance) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::solve: A == nullptr");
    }
    if (b == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::solve: b == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::BiCGSTAB::solve: x == nullptr");
    }

    // default iteration count to matrix size
    unsigned const iterations = maxIterations == 0 ? A->rows : maxIterations;

    // r0 = b - A * x0
    this->r->multiply(A, x, handle, stream);
    this->r->scalarMultiply(-1.0, stream);
    this->r->add(b, stream);

    // Choose an arbitrary vector rHat such that (r, rHat) != 0, e.g. rHat = r
    this->rHat->copy(r, stream);

    // initialize current error vector
    this->error->vectorDotProduct(this->r, this->r, stream);
    this->reference->vectorDotProduct(b, b, stream);
    this->error->copyToHost(stream);
    this->reference->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // iterate
    for (unsigned step = 0; step < iterations; ++step) {
        // roh = (rHat, r)
        this->rohOld->copy(this->roh, stream);
        this->roh->vectorDotProduct(this->r, this->rHat, stream);

        // beta = (roh(i) / roh(i-1)) * (alpha / omega)
        this->temp1->elementwiseDivision(this->roh, this->rohOld, stream);
        this->temp2->elementwiseDivision(this->alpha, this->omega, stream);
        this->beta->elementwiseMultiply(this->temp1, this->temp2, stream);

        // p = r + beta * (p - omega * nu)
        updateVector(this->p, -1.0, this->omega, this->nu, stream, this->temp1);
        updateVector(this->r, 1.0, this->beta, this->temp1, stream, this->p);

        // nu = A * p
        if (K != nullptr) {
            this->y->multiply(K, this->p, handle, stream);
        }
        else {
            this->y->copy(this->p, stream);
        }
        this->nu->multiply(A, this->y, handle, stream);

        // alpha = roh / (rHat, nu)
        this->temp1->vectorDotProduct(this->nu, this->rHat, stream);
        this->alpha->elementwiseDivision(this->roh, this->temp1, stream);

        // s = r - alpha * nu
        updateVector(this->r, -1.0, this->alpha, this->nu, stream, this->s);

        // t = A * s
        if (K != nullptr) {
            this->z->multiply(K, this->s, handle, stream);
        }
        else {
            this->z->copy(this->s, stream);
        }
        this->t->multiply(A, this->z, handle, stream);

        // omega = (t, s) / (t, t)
        if (K != nullptr) {
            this->temp2->multiply(K, this->t, handle, stream);
        }
        else {
            this->temp2->copy(this->t, stream);
        }
        this->temp1->vectorDotProduct(this->z, this->temp2, stream);
        this->temp2->vectorDotProduct(this->temp2, this->temp2, stream);
        this->omega->elementwiseDivision(this->temp1, this->temp2, stream);

        // x = x + alpha * p + omega * s
        updateVector(x, 1.0, this->alpha, this->y, stream, x);
        updateVector(x, 1.0, this->omega, this->z, stream, x);

        // r = s - omega * t
        updateVector(this->s, -1.0, this->omega, this->t, stream, this->r);

        // check error bound for all column vectors of residuum
        if (tolerance > 0.0) {
            using namespace std;
            using namespace thrust;

            this->error->vectorDotProduct(this->r, this->r, stream);
            this->error->copyToHost(stream);
            cudaStreamSynchronize(stream);

            for (unsigned i = 0; i < this->error->cols; ++i) {
                if (abs(sqrt((*this->error)(0, i) / (*this->reference)(0, i))) >= tolerance) {
                    break;
                }
                return step + 1;
            }
        }
    }

    return iterations;
}

// update vector
template <
    class dataType
>
void mpFlow::numeric::BiCGSTAB<dataType>::updateVector(
    std::shared_ptr<Matrix<dataType> const> const x1, double const sign,
    std::shared_ptr<Matrix<dataType> const> const x2,
    std::shared_ptr<Matrix<dataType> const> const scalar, cudaStream_t const stream,
    std::shared_ptr<Matrix<dataType>> const result) {
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
    dim3 blocks(result->dataRows / matrix::blockSize,
        result->dataCols == 1 ? 1 : result->dataCols / matrix::blockSize);
    dim3 threads(matrix::blockSize, result->dataCols == 1 ? 1 : matrix::blockSize);

    // execute kernel
    bicgstabKernel::updateVector(blocks, threads, stream, x1->deviceData, sign,
        x2->deviceData, scalar->deviceData, result->dataRows, result->deviceData);
}

// specialisations
template class mpFlow::numeric::BiCGSTAB<float>;
template class mpFlow::numeric::BiCGSTAB<double>;
template class mpFlow::numeric::BiCGSTAB<thrust::complex<float>>;
template class mpFlow::numeric::BiCGSTAB<thrust::complex<double>>;

template unsigned mpFlow::numeric::BiCGSTAB<float>::solve<mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<float>> const, std::shared_ptr<Matrix<float> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<float>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<float>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<float>::solve<mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const, std::shared_ptr<Matrix<float> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<float>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<double>::solve<mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<double>> const, std::shared_ptr<Matrix<double> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<double>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<double>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<double>::solve<mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const, std::shared_ptr<Matrix<double> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<double>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<thrust::complex<float>>::solve<mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>> const, std::shared_ptr<Matrix<thrust::complex<float>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<float>>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<thrust::complex<float>>::solve<mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const, std::shared_ptr<Matrix<thrust::complex<float>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<float>>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<thrust::complex<double>>::solve<mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>> const, std::shared_ptr<Matrix<thrust::complex<double>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<double>>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>> const, bool const, unsigned const, double const);
template unsigned mpFlow::numeric::BiCGSTAB<thrust::complex<double>>::solve<mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const, std::shared_ptr<Matrix<thrust::complex<double>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<double>>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const, bool const, unsigned const, double const);
