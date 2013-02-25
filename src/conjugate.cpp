// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"
#include "fasteit/conjugate_kernel.h"

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

// add scalar
void fastEIT::numeric::conjugate::addScalar(
    const std::shared_ptr<Matrix<dtype::real>> scalar,
    dtype::size rows, dtype::size columns, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> vector) {
    // check input
    if (scalar == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: scalar == nullptr");
    }
    if (vector == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: vector == nullptr");
    }

    // kernel dimension
    dim3 blocks(vector->data_rows() / matrix::block_size,
        vector->data_columns() == 1 ? 1 :
        vector->data_columns() / matrix::block_size);
    dim3 threads(matrix::block_size,
        vector->data_columns() == 1 ? 1 : matrix::block_size);

    // execute kernel
    conjugateKernel::addScalar(blocks, threads, stream, scalar->device_data(),
        vector->data_rows(), rows, columns, vector->device_data());
}

// update vector
void fastEIT::numeric::conjugate::updateVector(
    const std::shared_ptr<Matrix<dtype::real>> x1, dtype::real sign,
    const std::shared_ptr<Matrix<dtype::real>> x2,
    const std::shared_ptr<Matrix<dtype::real>> r1,
    const std::shared_ptr<Matrix<dtype::real>> r2, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> result) {
    // check input
    if (x1 == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: x1 == nullptr");
    }
    if (x2 == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: x2 == nullptr");
    }
    if (r1 == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: r1 == nullptr");
    }
    if (r2 == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: r2 == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: result == nullptr");
    }

    // kernel dimension
    dim3 blocks(result->data_rows() / matrix::block_size,
        result->data_columns() == 1 ? 1 :
        result->data_columns() / matrix::block_size);
    dim3 threads(matrix::block_size,
        result->data_columns() == 1 ? 1 : matrix::block_size);

    // execute kernel
    conjugateKernel::updateVector(blocks, threads, stream, x1->device_data(), sign,
        x2->device_data(), r1->device_data(), r2->device_data(), result->data_rows(),
        result->device_data());
}

// fast gemv
void fastEIT::numeric::conjugate::gemv(
    const std::shared_ptr<Matrix<dtype::real>> matrix,
    const std::shared_ptr<Matrix<dtype::real>> vector, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> result) {
    // check input
    if (matrix == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: matrix == nullptr");
    }
    if (vector == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: vector == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("Conjugate::addScalar: result == nullptr");
    }

    // dimension
    dim3 blocks(
        (matrix->data_rows() + 2 * matrix::block_size - 1) /
        (2 * matrix::block_size),
        ((matrix->data_rows() + 2 * matrix::block_size - 1 ) / (2 * matrix::block_size) +
        matrix::block_size - 1) / matrix::block_size);
    dim3 threads(2 * matrix::block_size, matrix::block_size);

    // call gemv kernel
    conjugateKernel::gemv(blocks, threads, stream, matrix->device_data(), vector->device_data(),
        matrix->data_rows(), result->device_data());

    // call reduce kernel
    conjugateKernel::reduceRow((matrix->data_columns() +
        matrix::block_size * matrix::block_size - 1) /
        (matrix::block_size * matrix::block_size),
        matrix::block_size * matrix::block_size, stream,
            result->data_rows(), result->device_data());
}
