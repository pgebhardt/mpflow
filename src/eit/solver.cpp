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
    template <class> class numericalSolverType,
    class equationType
>
mpFlow::EIT::Solver<numericalSolverType, equationType>::Solver(
    std::shared_ptr<equationType> const equation,
    std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
    unsigned const components, unsigned const parallelImages,
    dataType const regularizationFactor, cublasHandle_t const handle,
    cudaStream_t const stream) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: equation == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: handle == nullptr");
    }

    // create
    this->forwardSolver = std::make_shared<EIT::ForwardSolver<numericalSolverType, equationType>>(
        equation, source, components, handle, stream);

    // create inverse EIT
    this->inverseSolver = std::make_shared<solver::Inverse<dataType, numericalSolverType>>(
        equation->mesh->elements.rows(), forwardSolver->result->dataRows *
        forwardSolver->result->dataCols, parallelImages, regularizationFactor,
        handle, stream);

    // create matrices
    this->gamma = std::make_shared<numeric::Matrix<dataType>>(
        equation->mesh->elements.rows(), parallelImages, stream);
    this->dGamma = std::make_shared<numeric::Matrix<dataType>>(
        equation->mesh->elements.rows(), parallelImages, stream);
    for (unsigned image = 0; image < parallelImages; ++image) {
        this->measurement.push_back(std::make_shared<numeric::Matrix<dataType>>(
            source->measurementPattern->cols, source->drivePattern->cols, stream, 0.0, false));
        this->calculation.push_back(std::make_shared<numeric::Matrix<dataType>>(
            source->measurementPattern->cols, source->drivePattern->cols, stream, 0.0, false));
    }
}

// pre solve for accurate initial jacobian
template <
    template <class> class numericalSolverType,
    class equationType
>
void mpFlow::EIT::Solver<numericalSolverType, equationType>::preSolve(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::pre_solve: handle == nullptr");
    }

    // forward solving a few steps
    auto initialValue = this->forwardSolver->solve(this->gamma, handle, stream);

    // calc system matrix
    this->inverseSolver->calcSystemMatrix(this->forwardSolver->jacobian,
        solver::Inverse<dataType, numericalSolverType>::RegularizationType::square,
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
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::Solver<numericalSolverType, equationType>::solveDifferential(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solve_differential: handle == nullptr");
    }

    // solve
    this->inverseSolver->solve(this->forwardSolver->jacobian,
        this->calculation, this->measurement,
        this->forwardSolver->jacobian->cols / 8,
        handle, stream, this->dGamma);

    return this->dGamma;
}

// solve absolute
template <
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::Solver<numericalSolverType, equationType>::solveAbsolute(
    cublasHandle_t const handle, cudaStream_t const stream) {
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
    this->forwardSolver->solve(this->gamma, handle, stream);

    // calc inverse system matrix
    this->inverseSolver->calcSystemMatrix(this->forwardSolver->jacobian,
        solver::Inverse<dataType, numericalSolverType>::RegularizationType::square,
        handle, stream);

    // solve inverse
    this->inverseSolver->solve(this->forwardSolver->jacobian, { this->forwardSolver->result },
        this->measurement, this->forwardSolver->jacobian->cols / 8,
        handle, stream, this->dGamma);

    // add to gamma
    this->gamma->add(this->dGamma, stream);

    return this->gamma;
}

// specialisation
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
