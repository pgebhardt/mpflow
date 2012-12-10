// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <assert.h>

#include <stdexcept>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/sparse_matrix.h"
#include "../include/sparse_conjugate.h"
#include "../include/conjugate_cuda.h"

// create conjugate solver
fastEIT::numeric::SparseConjugate::SparseConjugate(dtype::size rows,
    dtype::size columns, cudaStream_t stream)
    : rows_(rows), columns_(columns) {
    // check input
    if (rows <= 1) {
        throw std::invalid_argument("SparseConjugate::SparseConjugate: rows <= 1");
    }
    if (columns <= 1) {
        throw std::invalid_argument("SparseConjugate::SparseConjugate: columns <= 1");
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
void fastEIT::numeric::SparseConjugate::solve(const std::shared_ptr<SparseMatrix> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations, bool dcFree,
    cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("SparseConjugate::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("SparseConjugate::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("SparseConjugate::solve: x == nullptr");
    }

    // calc residuum r = f - A * x
    A->multiply(x.get(), stream, this->residuum().get());

    // regularize for dc free solution
    if (dcFree == true) {
        this->temp_number()->sum(x.get(), stream);
        conjugate::addScalar(this->temp_number().get(), this->rows(), this->columns(),
            stream, this->residuum().get());
    }

    this->residuum()->scalarMultiply(-1.0, stream);
    this->residuum()->add(f.get(), stream);

    // p = r
    this->projection()->copy(this->residuum().get(), stream);

    // calc rsold
    this->rsold()->vectorDotProduct(this->residuum().get(), this->residuum().get(), stream);

    // iterate
    for (dtype::index step = 0; step < iterations; ++step) {
        // calc A * p
        A->multiply(this->projection().get(), stream, this->temp_vector().get());

        // regularize for dc free solution
        if (dcFree == true) {
            this->temp_number()->sum(this->projection().get(), stream);
            conjugate::addScalar(this->temp_number().get(), this->rows(), this->columns(),
                stream, this->temp_vector().get());
        }

        // calc p * A * p
        this->temp_number()->vectorDotProduct(this->projection().get(),
            this->temp_vector().get(), stream);

        // update residuum
        conjugate::updateVector(this->residuum().get(), -1.0f, this->temp_vector().get(),
            this->rsold().get(), this->temp_number().get(), stream, this->residuum().get());

        // update x
        conjugate::updateVector(x.get(), 1.0f, this->projection().get(), this->rsold().get(),
            this->temp_number().get(), stream, x.get());

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum().get(), this->residuum().get(), stream);

        // update projection
        conjugate::updateVector(this->residuum().get(), 1.0f, this->projection().get(),
            this->rsnew().get(), this->rsold().get(), stream, this->projection().get());

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew().get(), stream);
    }
}
