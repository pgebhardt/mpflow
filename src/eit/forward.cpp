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
    template <template <class> class> class numerical_solver
>
mpFlow::EIT::ForwardSolver<numerical_solver>::ForwardSolver(
    std::shared_ptr<mpFlow::EIT::model::Base> model, cudaStream_t stream)
    : model_(model) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::ForwardSolver: model == nullptr");
    }

    // create numerical_solver solver
    this->numeric_solver_ = std::make_shared<numerical_solver<mpFlow::numeric::SparseMatrix>>(
        this->model()->mesh()->nodes()->rows() + this->model()->electrodes()->count(),
        this->model()->source()->drive_count() + this->model()->source()->measurement_count(),
        stream);

    // create matrices
    this->voltage_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->model()->source()->measurement_count(),
        this->model()->source()->drive_count(), stream);
    this->current_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->model()->source()->measurement_count(),
        this->model()->source()->drive_count(), stream);
}

// apply pattern
template <
    template <template <class> class> class numerical_solver
>
void mpFlow::EIT::ForwardSolver<numerical_solver>::applyMeasurementPattern(
    std::shared_ptr<numeric::Matrix<dtype::real>> result, cudaStream_t stream) {
    // check input
    if (result == nullptr) {
        throw std::invalid_argument("forward::applyPattern: result == nullptr");
    }

    // apply pattern
    dim3 threads(result->rows(), result->columns());
    dim3 blocks(1, 1);

    forwardKernel::applyMeasurementPattern(blocks, threads, stream,
        this->model()->potential(0)->device_data(),
        this->model()->mesh()->nodes()->rows(),
        this->model()->potential(0)->data_rows(),
        this->model()->source()->measurement_pattern()->device_data(),
        this->model()->source()->measurement_pattern()->data_rows(),
        false, result->device_data(), result->data_rows());

    for (dtype::index component = 1; component < this->model()->component_count(); ++component) {
        forwardKernel::applyMeasurementPattern(blocks, threads, stream,
            this->model()->potential(component)->device_data(),
            this->model()->mesh()->nodes()->rows(),
            this->model()->potential(component)->data_rows(),
            this->model()->source()->measurement_pattern()->device_data(),
            this->model()->source()->measurement_pattern()->data_rows(),
            true, result->device_data(), result->data_rows());
    }
}

// forward solving
template <
    template <template <class> class> class numerical_solver
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> mpFlow::EIT::ForwardSolver<numerical_solver>::solve(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, dtype::size steps, cudaStream_t stream) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: gamma == nullptr");
    }

    // update system matrix
    this->model()->update(gamma, stream);

    // solve for ground mode
    this->numeric_solver()->solve(this->model()->system_matrix(0),
        this->model()->source()->excitation(0), steps,
        this->model()->source()->type() == "current" ? true : false,
        nullptr, stream, this->model()->potential(0));

    // solve for higher harmonics
    for (dtype::index component = 1; component < this->model()->component_count(); ++component) {
        this->numeric_solver()->solve(
            this->model()->system_matrix(component),
            this->model()->source()->excitation(component),
            steps, false, nullptr, stream, this->model()->potential(component));
    }

    // calc jacobian
    this->model()->calcJacobian(gamma, stream);

    // current source specific tasks
    if (this->model()->source()->type() == "current") {
        // calc voltage
        this->applyMeasurementPattern(this->voltage(), stream);

        return this->voltage();
    } else if (this->model()->source()->type() == "voltage") {
        // calc current
        this->applyMeasurementPattern(this->current(), stream);

        // scale current with electrode height
        this->current()->scalarMultiply(std::get<1>(this->model()->electrodes()->shape()), stream);

        return this->current();
    }

    // default result
    return this->voltage();
}

// specialisation
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient>;
