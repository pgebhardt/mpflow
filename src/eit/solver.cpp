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

template <
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::Solver(
    std::shared_ptr<equationType> const equation,
    std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
    unsigned const components, unsigned const parallelImages,
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: equation == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::Solver: handle == nullptr");
    }

    // create forward solver
    this->forwardSolver = std::make_shared<EIT::ForwardSolver<numericalForwardSolverType, equationType>>(
        equation, source, components, handle, stream);

    // create inverse solver
    this->inverseSolver = std::make_shared<solver::Inverse<dataType, numericalInverseSolverType>>(
        equation->mesh->elements.rows(), forwardSolver->result->dataRows *
        forwardSolver->result->dataCols, parallelImages, handle, stream);

    // create matrices
    this->result = std::make_shared<numeric::Matrix<dataType>>(
        equation->mesh->elements.rows(), parallelImages, stream);
    this->delta = std::make_shared<numeric::Matrix<dataType>>(
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
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
void mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::preSolve(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Solver::pre_solve: handle == nullptr");
    }

    // forward solving a few steps
    auto const initialDistribution = std::make_shared<numeric::Matrix<dataType>>(
        this->result->rows, this->result->cols, stream,
        equationType::logarithmic ? dataType(0) : this->forwardSolver->equation->referenceValue);
    auto const initialValue = this->forwardSolver->solve(initialDistribution, handle, stream);

    // calc initial system matrix
    this->inverseSolver->updateJacobian(this->forwardSolver->jacobian, handle, stream);

    // set measurement and calculation to initial value of forward model
    for (auto level : this->measurement) {
        level->copy(initialValue, stream);
    }
    for (auto level : this->calculation) {
        level->copy(initialValue, stream);
    }
}

// solve differential
template <
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::solveDifferential(
    cublasHandle_t const handle, cudaStream_t const stream, unsigned const maxIterations,
    unsigned* const iterations) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solve_differential: handle == nullptr");
    }

    // solve
    unsigned const steps = this->inverseSolver->solve(this->calculation, this->measurement,
        maxIterations, handle, stream, this->result);

    if (iterations != nullptr) {
        *iterations = steps;
    }

    return this->result;
}

// solve absolute
template <
    template <class> class numericalForwardSolverType,
    template <class> class numericalInverseSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::Solver<numericalForwardSolverType, numericalInverseSolverType, equationType>::solveAbsolute(
    unsigned const iterations, cublasHandle_t const handle, cudaStream_t const stream) {
    // only execute method, when parallelImages == 1
    if (this->measurement.size() != 1) {
        throw std::runtime_error(
            "mpFlow::EIT::Solver::solveAbsolute: parallelImages != 1");
    }

    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Solver::solveAbsolute: handle == nullptr");
    }

    // initialize with homogeneous material distribution
    this->result->fill(equationType::logarithmic ? dataType(0) :
        this->forwardSolver->equation->referenceValue, stream);
    
    // do newton iterations
    for (unsigned step = 0; step < iterations; ++step) {
        // solve for new jacobian and reference data
        this->forwardSolver->solve(this->result, handle, stream);
        this->inverseSolver->updateJacobian(this->forwardSolver->jacobian, handle, stream);
    
        // solve inverse
        this->inverseSolver->solve({ this->forwardSolver->result },
            this->measurement, 0, handle, stream, this->delta);
    
        // add to result
        this->result->add(this->delta, stream);
    }
    
    return this->result;
}

// specialisation
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;

template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
        
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;

template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::ConjugateGradient, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
    template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::Solver<mpFlow::numeric::BiCGSTAB, mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;