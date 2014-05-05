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
    cublasHandle_t handle, cudaStream_t stream)
    : equation(equation), source(source) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::ForwardSolver::ForwardSolver: equation == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::ForwardSolver::ForwardSolver: handle == nullptr");
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
     this->electrodesAttachmentMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->source->measurementPattern->columns(),
        this->equation->mesh->nodes()->rows(), stream);

    // calc electrodes attachement matrix
    cublasSetStream(handle, stream);
    dtype::real alpha = 1.0, beta = 0.0;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        this->source->measurementPattern->data_columns(),
        this->equation->excitationMatrix->data_rows(),
        this->source->measurementPattern->data_rows(), &alpha,
        this->source->measurementPattern->device_data(),
        this->source->measurementPattern->data_rows(),
        this->equation->excitationMatrix->device_data(),
        this->equation->excitationMatrix->data_rows(),
        &beta, this->electrodesAttachmentMatrix->device_data(),
        this->electrodesAttachmentMatrix->data_rows()) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::EIT::ForwardSolver: calc voltage calculation");
    }
}

// apply pattern
template <
    class equationType,
    template <template <class> class> class numericalSolverType
>
void mpFlow::EIT::ForwardSolver<equationType, numericalSolverType>::applyMeasurementPattern(
    std::shared_ptr<numeric::Matrix<dtype::real>> result, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (result == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMeasurementPattern: result == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMeasurementPattern: handle == nullptr");
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    dtype::real alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->electrodesAttachmentMatrix->data_rows(),
        this->source->drivePattern->columns(), this->electrodesAttachmentMatrix->data_columns(), &alpha,
        this->electrodesAttachmentMatrix->device_data(), this->electrodesAttachmentMatrix->data_rows(),
        this->phi->device_data(), this->phi->data_rows(), &beta,
        result->device_data(), result->data_rows());
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
    this->excitation->multiply(this->equation->excitationMatrix,
        this->source->pattern, handle, stream);

    // solve for ground mode
    this->numericalSolver->solve(this->equation->systemMatrix,
        this->excitation, steps,
        this->source->type == mpFlow::EIT::source::CurrentSourceType ? true : false,
        nullptr, stream, this->phi);

    // calc jacobian
    this->equation->calcJacobian(this->phi, gamma, this->source->drivePattern->columns(),
        this->source->measurementPattern->columns(), stream, this->jacobian);

    // current source specific tasks
    if (this->source->type == source::CurrentSourceType) {
        this->jacobian->scalarMultiply(-1.0, stream);

        // calc voltage
        this->applyMeasurementPattern(this->voltage, handle, stream);

        return this->voltage;
    }
    else if (this->source->type == source::VoltageSourceType) {
        // calc current
        this->applyMeasurementPattern(this->current, handle, stream);

        return this->current;
    }

    // default result
    return this->voltage;
}

// specialisation
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::EllipticalEquation<mpFlow::FEM::basis::Linear>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::EllipticalEquation<mpFlow::FEM::basis::Quadratic>, mpFlow::numeric::ConjugateGradient>;
