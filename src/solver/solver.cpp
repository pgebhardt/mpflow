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

// create solver
template <
    class forward_solver_type,
    template <template <class> class> class numerical_inverse_solver_type
>
mpFlow::solver::Solver<forward_solver_type, numerical_inverse_solver_type>::Solver(
    std::shared_ptr<forward_solver_type> forward_solver, dtype::index parallel_images,
    dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream)
    : forward_solver_(nullptr), inverse_solver_(nullptr), gamma_(nullptr),
    dgamma_(nullptr) {
    // check input
    if (forward_solver == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Solver::Solver: forward_solver == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Solver::Solver: handle == nullptr");
    }

    // save forward solver
    this->forward_solver_ = forward_solver;

    // create inverse solver
    this->inverse_solver_ = std::make_shared<Inverse<numerical_inverse_solver_type>>(
        this->forward_solver()->equation->mesh->elements()->rows(),
        math::roundTo(this->forward_solver()->source->measurementPattern->columns(),
            numeric::matrix::block_size) *
        math::roundTo(this->forward_solver()->source->drivePattern->columns(),
            numeric::matrix::block_size),
        parallel_images, regularization_factor, handle, stream);

    // create matrices
    this->gamma_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->forward_solver()->equation->mesh->elements()->rows(), parallel_images, stream);
    this->dgamma_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->forward_solver()->equation->mesh->elements()->rows(), parallel_images, stream);
    for (dtype::index image = 0; image < parallel_images; ++image) {
        this->measurement_.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            this->forward_solver()->source->measurementPattern->columns(),
            this->forward_solver()->source->drivePattern->columns(), stream));
        this->calculation_.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            this->forward_solver()->source->measurementPattern->columns(),
            this->forward_solver()->source->drivePattern->columns(), stream));
    }
}

// pre solve for accurate initial jacobian
template <
    class forward_solver_type,
    template <template <class> class> class numerical_inverse_solver_type
>
void mpFlow::solver::Solver<forward_solver_type, numerical_inverse_solver_type>::preSolve(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Solver::pre_solve: handle == nullptr");
    }

    // forward solving a few steps
    auto initial_value = this->forward_solver()->solve(this->gamma(),
        this->forward_solver()->equation->mesh->nodes()->rows() / 4,
        handle, stream);

    // calc system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->jacobian,
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
    class forward_solver_type,
    template <template <class> class> class numerical_inverse_solver_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::solver::Solver<forward_solver_type, numerical_inverse_solver_type>::solve_differential(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::solver::Solver::solve_differential: handle == nullptr");
    }

    // solve
    this->inverse_solver()->solve(this->forward_solver()->jacobian,
        this->calculation(), this->measurement(),
        this->forward_solver()->equation->mesh->elements()->rows() / 8,
        handle, stream, this->dgamma());

    return this->dgamma();
}

// solve absolute
template <
    class forward_solver_type,
    template <template <class> class> class numerical_inverse_solver_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::solver::Solver<forward_solver_type, numerical_inverse_solver_type>::solve_absolute(
    cublasHandle_t handle, cudaStream_t stream) {
    // only execute method, when parallel_images == 1
    if (this->measurement().size() != 1) {
        throw std::runtime_error(
            "mpFlow::solver::Solver::solve_absolute: parallel_images != 1");
    }

    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::solver::Solver::solve_absolute: handle == nullptr");
    }

    // solve forward
    this->forward_solver()->solve(this->gamma(),
        this->forward_solver()->equation->mesh->nodes()->rows() / 80,
        handle, stream);

    // calc inverse system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->jacobian,
        handle, stream);

    // solve inverse
    std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> calculation(
        1, this->forward_solver()->voltage);
    this->inverse_solver()->solve(this->forward_solver()->jacobian, calculation,
        this->measurement(), this->forward_solver()->equation->mesh->elements()->rows() / 8,
        handle, stream, this->dgamma());

    // add to gamma
    this->gamma()->add(this->dgamma(), stream);

    return this->gamma();
}

// specialisation
template class mpFlow::solver::Solver<
    mpFlow::EIT::ForwardSolver<
        mpFlow::FEM::EllipticalEquation<mpFlow::FEM::basis::Linear>,
        mpFlow::numeric::ConjugateGradient>,
    mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Solver<
    mpFlow::EIT::ForwardSolver<
        mpFlow::FEM::EllipticalEquation<mpFlow::FEM::basis::Quadratic>,
        mpFlow::numeric::ConjugateGradient>,
    mpFlow::numeric::ConjugateGradient>;
