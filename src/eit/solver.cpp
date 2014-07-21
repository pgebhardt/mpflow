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

// create EIT
template <
    class basisFunctionType
>
mpFlow::EIT::Solver<basisFunctionType>::Solver(std::shared_ptr<mpFlow::EIT::Equation<basisFunctionType>> equation,
    std::shared_ptr<mpFlow::FEM::SourceDescriptor> source, dtype::index components,
    dtype::index parallelImages, dtype::real regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: equation == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: handle == nullptr");
    }

    // create
    this->forwardSolver = std::make_shared<EIT::ForwardSolver<basisFunctionType, numeric::ConjugateGradient>>(
        equation, source, components, handle, stream);

    // create inverse EIT
    this->inverseSolver = std::make_shared<InverseSolver<numeric::ConjugateGradient>>(
        equation->mesh->elements()->rows(), forwardSolver->voltage->data_rows() *
        forwardSolver->voltage->data_columns(), parallelImages, regularizationFactor,
        handle, stream);

    // create matrices
    this->gamma = std::make_shared<numeric::Matrix<dtype::real>>(
        equation->mesh->elements()->rows(), parallelImages, stream);
    this->dGamma = std::make_shared<numeric::Matrix<dtype::real>>(
        equation->mesh->elements()->rows(), parallelImages, stream);
    for (dtype::index image = 0; image < parallelImages; ++image) {
        this->measurement.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            source->measurementPattern->columns(), source->drivePattern->columns(), stream));
        this->calculation.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            source->measurementPattern->columns(), source->drivePattern->columns(), stream));
    }
}

// pre solve for accurate initial jacobian
template <
    class basisFunctionType
>
void mpFlow::EIT::Solver<basisFunctionType>::preSolve(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::pre_solve: handle == nullptr");
    }

    // forward solving a few steps
    auto initialValue = this->forwardSolver->solve(this->gamma,
        this->forwardSolver->equation->mesh->nodes()->rows() / 4,
        handle, stream);

    // calc system matrix
    this->inverseSolver->calcSystemMatrix(this->forwardSolver->jacobian,
        handle, stream);

    // set measurement and calculation to initial value of forward EIT
    for (auto level : this->measurement) {
        level->copy(initialValue, stream);
    }
    for (auto level : this->calculation) {
        level->copy(initialValue, stream);
    }
}

// solve differential
template <
    class basisFunctionType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::Solver<basisFunctionType>::solveDifferential(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solve_differential: handle == nullptr");
    }

    // solve
    this->inverseSolver->solve(this->forwardSolver->jacobian,
        this->calculation, this->measurement,
        this->forwardSolver->equation->mesh->elements()->rows() / 8,
        handle, stream, this->dGamma);

    return this->dGamma;
}

// solve absolute
template <
    class basisFunctionType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::Solver<basisFunctionType>::solveAbsolute(
    cublasHandle_t handle, cudaStream_t stream) {
    // only execute method, when parallel_images == 1
    if (this->measurement.size() != 1) {
        throw std::runtime_error(
            "mpFlow::EIT::Solver::solve_absolute: parallel_images != 1");
    }

    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solve_absolute: handle == nullptr");
    }

    // solve forward
    this->forwardSolver->solve(this->gamma,
        this->forwardSolver->equation->mesh->nodes()->rows() / 80,
        handle, stream);

    // calc inverse system matrix
    this->inverseSolver->calcSystemMatrix(this->forwardSolver->jacobian,
        handle, stream);

    // solve inverse
    std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> calculation(
        1, this->forwardSolver->voltage);
    this->inverseSolver->solve(this->forwardSolver->jacobian, calculation,
        this->measurement, this->forwardSolver->equation->mesh->elements()->rows() / 8,
        handle, stream, this->dGamma);

    // add to gamma
    this->gamma->add(this->dGamma, stream);

    return this->gamma;
}

// specialisation
template class mpFlow::EIT::Solver<mpFlow::FEM::basis::Linear>;
template class mpFlow::EIT::Solver<mpFlow::FEM::basis::Quadratic>;
