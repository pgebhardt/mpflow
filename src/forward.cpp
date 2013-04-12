// fastEIT
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"
#include "fasteit/forward_kernel.h"

// create forward_solver
template <
    class numeric_solver_type
>
fastEIT::ForwardSolver<numeric_solver_type>::ForwardSolver(
    std::shared_ptr<fastEIT::model::Model> model, cublasHandle_t handle, cudaStream_t stream)
    : model_(model) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::ForwardSolver: model == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::ForwardSolver: handle == nullptr");
    }

    // create numeric_solver_type solver
    this->numeric_solver_ = std::make_shared<numeric_solver_type>(
        this->model()->mesh()->nodes()->rows() + this->model()->electrodes()->count(),
        this->model()->source()->drive_count() + this->model()->source()->measurement_count(),
        stream);

    // create matrices
    this->voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->source()->measurement_count(),
        this->model()->source()->drive_count(), stream);
    this->current_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->source()->measurement_count(),
        this->model()->source()->drive_count(), stream);
}

// apply pattern
template <
    class numeric_solver_type
>
void fastEIT::ForwardSolver<numeric_solver_type>::applyMeasurementPattern(
    std::shared_ptr<Matrix<dtype::real>> result, cudaStream_t stream) {
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
    class numeric_solver_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::ForwardSolver<numeric_solver_type>::solve(
    const std::shared_ptr<Matrix<dtype::real>> gamma, dtype::size steps, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::solve: handle == nullptr");
    }

    // update system matrix
    this->model()->update(gamma, handle, stream);

    // solve for ground mode
    this->numeric_solver()->solve(this->model()->system_matrix(0),
        this->model()->source()->excitation(0), steps,
        this->model()->source()->type() == "current" ? true : false,
        stream, this->model()->potential(0));

    // solve for higher harmonics
    for (dtype::index component = 1; component < this->model()->component_count(); ++component) {
        this->numeric_solver()->solve(
            this->model()->system_matrix(component),
            this->model()->source()->excitation(component),
            steps, false, stream, this->model()->potential(component));
    }

    // calc jacobian
    this->model()->calcJacobian(gamma, stream);

    // current source specific tasks
    if (this->model()->source()->type() == "current") {
        // turn sign of jacobian, because of voltage jacobian
        this->model()->jacobian()->scalarMultiply(-1.0, stream);

        // calc voltage
        this->applyMeasurementPattern(this->voltage(), stream);

        return this->voltage();
    } else if (this->model()->source()->type() == "voltage") {
        // calc current
        this->applyMeasurementPattern(this->current(), stream);

        return this->current();
    }

    // default result
    return this->voltage();
}

// specialisation
template class fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate>;
