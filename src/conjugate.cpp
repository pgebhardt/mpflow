// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"
#include "../include/conjugate_cuda.h"

// create conjugate solver
fastEIT::numeric::Conjugate::Conjugate(dtype::size rows, cublasHandle_t handle,
    cudaStream_t stream)
    : rows_(rows)  {
    // check input
    if (rows <= 1) {
        throw std::invalid_argument("Conjugate::Conjugate: rows <= 1");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Conjugate::Conjugate: handle == NULL");
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
void fastEIT::numeric::Conjugate::solve(const std::shared_ptr<Matrix<dtype::real>> A,
    const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations,
    cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> x) {
    // check input
    if (A == nullptr) {
        throw std::invalid_argument("Conjugate::solve: A == nullptr");
    }
    if (f == nullptr) {
        throw std::invalid_argument("Conjugate::solve: f == nullptr");
    }
    if (x == nullptr) {
        throw std::invalid_argument("Conjugate::solve: x == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Conjugate::solve: handle == NULL");
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
        conjugate::gemv(A.get(), this->projection().get(), stream, this->temp_vector().get());

        // calc p * A * p
        this->temp_number()->vectorDotProduct(this->projection(),
            this->temp_vector(), stream);

        // update residuum
        conjugate::updateVector(this->residuum().get(), -1.0f, this->temp_vector().get(),
            this->rsold().get(), this->temp_number().get(), stream, this->residuum().get());

        // update x
        conjugate::updateVector(x.get(), 1.0f, this->projection().get(), this->rsold().get(),
            this->temp_number().get(), stream, x.get());

        // calc rsnew
        this->rsnew()->vectorDotProduct(this->residuum(), this->residuum(), stream);

        // update projection
        conjugate::updateVector(this->residuum().get(), 1.0f, this->projection().get(),
            this->rsnew().get(), this->rsold().get(), stream, this->projection().get());

        // copy rsnew to rsold
        this->rsold()->copy(this->rsnew(), stream);
    }
}

