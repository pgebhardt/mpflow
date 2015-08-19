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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"

using namespace std;
using namespace thrust;

// solve conjugateGradient spars conste
template <
    class dataType
>
template <
    template <class> class matrixType,
    template <class> class preconditionerType
>
unsigned mpFlow::numeric::CPUSolver<dataType>::solve(std::shared_ptr<matrixType<dataType>> const A,
    std::shared_ptr<Matrix<dataType> const> const b, cublasHandle_t const,
    cudaStream_t const stream, std::shared_ptr<Matrix<dataType>> const x,
    std::shared_ptr<preconditionerType<dataType>> const, unsigned const, bool const) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::CPUSolver::solve: A == nullptr");
    }
    if (b == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::CPUSolver::solve: b == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("mpFlow::numeric::CPUSolver::solve: x == nullptr");
    }

    // create eigen versions of all parameter matrices
    A->copyToHost(stream);
    b->copyToHost(stream);
    cudaStreamSynchronize(stream);
    
    auto const S = A->toEigen(stream).matrix().eval();
    auto const f = b->toEigen(stream).matrix().eval();
    
    // solve system
    auto const res = S.partialPivLu().solve(f).eval();
    
    x->copy(numeric::Matrix<dataType>::fromEigen(res.array(), stream), stream);
        
    return 0;
}

// specialisations
template class mpFlow::numeric::CPUSolver<float>;
template class mpFlow::numeric::CPUSolver<double>;
template class mpFlow::numeric::CPUSolver<thrust::complex<float>>;
template class mpFlow::numeric::CPUSolver<thrust::complex<double>>;

template unsigned mpFlow::numeric::CPUSolver<float>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<float>> const, std::shared_ptr<Matrix<float> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<float>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<float>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<float>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<float>> const, std::shared_ptr<Matrix<float> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<float>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<float>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const, std::shared_ptr<Matrix<float> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<float>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<float>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<float>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const, std::shared_ptr<Matrix<float> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<float>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<float>> const, unsigned const, bool const);

template unsigned mpFlow::numeric::CPUSolver<double>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<double>> const, std::shared_ptr<Matrix<double> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<double>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<double>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<double>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<double>> const, std::shared_ptr<Matrix<double> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<double>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<double>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const, std::shared_ptr<Matrix<double> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<double>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<double>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<double>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const, std::shared_ptr<Matrix<double> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<double>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<double>> const, unsigned const, bool const);

template unsigned mpFlow::numeric::CPUSolver<thrust::complex<float>>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>> const, std::shared_ptr<Matrix<thrust::complex<float>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<float>>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<thrust::complex<float>>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>> const, std::shared_ptr<Matrix<thrust::complex<float>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<float>>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<thrust::complex<float>>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const, std::shared_ptr<Matrix<thrust::complex<float>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<float>>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<thrust::complex<float>>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const, std::shared_ptr<Matrix<thrust::complex<float>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<float>>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<float>>> const, unsigned const, bool const);

template unsigned mpFlow::numeric::CPUSolver<thrust::complex<double>>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>> const, std::shared_ptr<Matrix<thrust::complex<double>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<double>>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<thrust::complex<double>>::solve<mpFlow::numeric::Matrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>> const, std::shared_ptr<Matrix<thrust::complex<double>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<double>>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<thrust::complex<double>>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::Matrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const, std::shared_ptr<Matrix<thrust::complex<double>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<double>>> const,
    std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<double>>> const, unsigned const, bool const);
template unsigned mpFlow::numeric::CPUSolver<thrust::complex<double>>::solve<mpFlow::numeric::SparseMatrix, mpFlow::numeric::SparseMatrix>(
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const, std::shared_ptr<Matrix<thrust::complex<double>> const> const,
    cublasHandle_t const, cudaStream_t const, std::shared_ptr<Matrix<thrust::complex<double>>> const,
    std::shared_ptr<mpFlow::numeric::SparseMatrix<thrust::complex<double>>> const, unsigned const, bool const);