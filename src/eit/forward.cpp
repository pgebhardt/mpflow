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
#include "mpflow/eit/forward_kernel.h"

// create forward_solver
template <
    class equationType,
    template <template <class> class> class numericalSolverType
>
mpFlow::EIT::ForwardSolver<equationType, numericalSolverType>::ForwardSolver(
    std::shared_ptr<equationType> equation, std::shared_ptr<Source> source,
    cudaStream_t stream)
    : equation(equation), source(source) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::ForwardSolver::ForwardSolver: equation == nullptr");
    }

    // create numericalSolver solver
    this->numericalSolver = std::make_shared<numericalSolverType<
        mpFlow::numeric::SparseMatrix>>(
        this->equation->mesh->nodes()->rows(),
        this->source->drivePattern->columns() + this->source->measurementPattern->columns(), stream);

    // create matrices
    this->voltage = std::make_shared<numeric::Matrix<dtype::real>>(
        this->source->measurementPattern->columns(), this->source->drivePattern->columns(), stream);
    this->current = std::make_shared<numeric::Matrix<dtype::real>>(
        this->source->measurementPattern->columns(), this->source->drivePattern->columns(), stream);
    this->phi = std::make_shared<numeric::Matrix<dtype::real>>(this->equation->mesh->nodes()->rows(),
        this->source->pattern->columns(), stream);
    this->excitation = std::make_shared<numeric::Matrix<dtype::real>>(this->equation->mesh->nodes()->rows(),
        this->source->pattern->columns(), stream);
    this->jacobian = std::make_shared<numeric::Matrix<dtype::real>>(
        math::roundTo(this->source->measurementPattern->columns(), numeric::matrix::block_size) *
        math::roundTo(this->source->drivePattern->columns(), numeric::matrix::block_size),
        this->equation->mesh->elements()->rows(), stream);
}

// apply pattern
template <
    class equationType,
    template <template <class> class> class numericalSolverType
>
void mpFlow::EIT::ForwardSolver<equationType, numericalSolverType>::applyMeasurementPattern(
    std::shared_ptr<numeric::Matrix<dtype::real>> result, cudaStream_t) {
    // check input
    if (result == nullptr) {
        throw std::invalid_argument("forward::applyPattern: result == nullptr");
    }

    // TODO: applyMeasurementPattern
}

// forward solving
template <
    class equationType,
    template <template <class> class> class numericalSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::ForwardSolver<equationType, numericalSolverType>::solve(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: handle == nullptr");
    }

    // update system matrix
    this->equation->update(gamma, 0.0, stream);
    this->source->updateExcitation(this->excitation, handle, stream);
    this->excitation->multiply(this->equation->excitationMatrix, this->source->pattern,
        handle, stream);

    // solve for ground mode
    this->numericalSolver->solve(this->equation->systemMatrix,
        this->excitation, steps,
        this->source->type == mpFlow::EIT::source::CurrentSourceType ? true : false,
        nullptr, stream, this->phi);
/*
    // solve for higher harmonics
    for (dtype::index component = 1; component < this->equation->component_count(); ++component) {
        this->numericalSolver->solve(
            this->equation->system_matrix(component),
            this->equation->source()->excitation(component),
            steps, false, nullptr, stream, this->equation->potential(component));
    }
*/
    // calc jacobian
    this->calcJacobian(gamma, stream);

    // current source specific tasks
    if (this->source->type == "current") {
        // calc voltage
        this->applyMeasurementPattern(this->voltage, stream);

        return this->voltage;
    }
    else if (this->source->type == "voltage") {
        // calc current
        this->applyMeasurementPattern(this->current, stream);

        // scale current with electrode height
        // this->current()->scalarMultiply(std::get<1>(this->equation->electrodes()->shape()), stream);

        return this->current;
    }

    // default result
    return this->voltage;
}

// calc jacobian
template <
    class equationType,
    template <template <class> class> class numericalSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::ForwardSolver<equationType, numericalSolverType>::calcJacobian(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, cudaStream_t stream) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::calcJacobian: gamma == nullptr");
    }

    // calc jacobian
    this->equation->calcJacobian(this->phi, gamma, this->source->drivePattern->columns(),
        this->source->measurementPattern->columns(), stream, this->jacobian);
/*    for (dtype::index component = 1; component < this->component_count(); ++component) {
        ellipticalEquation::calcJacobian<basisFunctionType>(
            gamma, this->potential(component), this->mesh()->elements(),
            this->elemental_jacobian_matrix(), this->source()->drive_count(),
            this->source()->measurement_count(), this->sigma_ref(),
            true, stream, this->jacobian());
    }*/

    // switch sign if current source
    if (this->source->type == "current") {
        this->jacobian->scalarMultiply(-1.0, stream);
    }

    return this->jacobian;
}

// specialisation
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::EllipticalEquation<mpFlow::FEM::basis::Linear>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::EllipticalEquation<mpFlow::FEM::basis::Quadratic>, mpFlow::numeric::ConjugateGradient>;
