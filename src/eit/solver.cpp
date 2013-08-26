// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "mpflow/mpflow.h"

// create solver
template <
    class numerical_inverse_solver_type
>
mpFlow::EIT::solver::Solver<numerical_inverse_solver_type>::Solver(
    std::shared_ptr<model::Base> model, dtype::index parallel_images,
    dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream)
    : forward_solver_(nullptr), inverse_solver_(nullptr), gamma_(nullptr),
    dgamma_(nullptr) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::solver::Solver::Solver: model == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::solver::Solver::Solver: handle == nullptr");
    }

    // create forward solver
    this->forward_solver_ = std::make_shared<Forward<numeric::SparseConjugate>>(
        model, stream);

    // create inverse solver
    this->inverse_solver_ = std::make_shared<Inverse<numerical_inverse_solver_type>>(
        this->forward_solver()->model()->mesh()->elements()->rows(),
        math::roundTo(this->forward_solver()->model()->source()->measurement_count(),
            numeric::matrix::block_size) *
        math::roundTo(this->forward_solver()->model()->source()->drive_count(),
            numeric::matrix::block_size),
        parallel_images, regularization_factor, handle, stream);

    // create matrices
    this->gamma_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->forward_solver()->model()->mesh()->elements()->rows(), parallel_images, stream);
    this->dgamma_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->forward_solver()->model()->mesh()->elements()->rows(), parallel_images, stream);
    for (dtype::index image = 0; image < parallel_images; ++image) {
        this->measurement_.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            this->forward_solver()->model()->source()->measurement_count(),
            this->forward_solver()->model()->source()->drive_count(), stream));
        this->calculation_.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            this->forward_solver()->model()->source()->measurement_count(),
            this->forward_solver()->model()->source()->drive_count(), stream));
    }
}

// pre solve for accurate initial jacobian
template <
    class numerical_inverse_solver_type
>
void mpFlow::EIT::solver::Solver<numerical_inverse_solver_type>::preSolve(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::solver::Solver::pre_solve: handle == nullptr");
    }

    // forward solving a few steps
    auto initial_value = this->forward_solver()->solve(this->gamma(),
        this->forward_solver()->model()->mesh()->nodes()->rows() / 4, stream);

    // calc system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->model()->jacobian(),
        handle, stream);

    // set measurement and calculation to initial value of forward solver
    for (auto level : this->measurement_) {
        level->copy(initial_value, stream);
    }
    for (auto level : this->calculation_) {
        level->copy(initial_value, stream);
    }
}

// solve differential
template <
    class numerical_inverse_solver_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::solver::Solver<numerical_inverse_solver_type>::solve_differential(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::solver::Solver::solve_differential: handle == nullptr");
    }

    // solve
    this->inverse_solver()->solve(this->forward_solver()->model()->jacobian(),
        this->calculation(), this->measurement(),
        this->forward_solver()->model()->mesh()->elements()->rows() / 8,
        handle, stream, this->dgamma());

    return this->dgamma();
}

// solve absolute
template <
    class numerical_inverse_solver_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::solver::Solver<numerical_inverse_solver_type>::solve_absolute(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::solver::Solver::solve_absolute: handle == nullptr");
    }

    // solve forward
    this->forward_solver()->solve(this->gamma(),
        this->forward_solver()->model()->mesh()->nodes()->rows() / 80,  stream);

    // calc inverse system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->model()->jacobian(),
        handle, stream);

    // solve inverse
    std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> calculation(
        1, this->forward_solver()->voltage());
    this->inverse_solver()->solve(this->forward_solver()->model()->jacobian(), calculation,
        this->measurement(), this->forward_solver()->model()->mesh()->elements()->rows() / 8,
        handle, stream, this->dgamma());

    // add to gamma
    this->gamma()->add(this->dgamma(), stream);

    return this->gamma();
}

// specialisation
template class mpFlow::EIT::solver::Solver<mpFlow::numeric::Conjugate>;
template class mpFlow::EIT::solver::Solver<mpFlow::numeric::FastConjugate>;
