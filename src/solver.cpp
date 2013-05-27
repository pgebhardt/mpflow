// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"

// create solver
fastEIT::Solver::Solver(std::shared_ptr<fastEIT::model::Model> model,
    dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream)
    : model_(model) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::Solver: model == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::Solver: handle == nullptr");
    }

    // create forward solver
    this->forward_solver_ = std::make_shared<ForwardSolver<numeric::SparseConjugate>>(
        this->model(), handle, stream);

    // create inverse solver
    this->inverse_solver_ = std::make_shared<InverseSolver<numeric::Conjugate>>(
        this->model()->mesh()->elements()->rows(),
        math::roundTo(this->model()->source()->measurement_count(), matrix::block_size) *
        math::roundTo(this->model()->source()->drive_count(), matrix::block_size),
        regularization_factor, handle, stream);

    // create matrices
    this->dgamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), 16, stream);
    this->gamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), 16, stream);
    this->measured_voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->source()->measurement_count(),
        this->model()->source()->drive_count(), stream);
    this->calibration_voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->source()->measurement_count(),
        this->model()->source()->drive_count(), stream);
}

// pre solve for accurate initial jacobian
void fastEIT::Solver::preSolve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::pre_solve: handle == nullptr");
    }

    // forward solving a few steps
    this->forward_solver()->solve(this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverse_solver()->calcSystemMatrix(this->model()->jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    this->measured_voltage()->copy(this->forward_solver()->voltage(), stream);
    this->calibration_voltage()->copy(this->forward_solver()->voltage(), stream);
}

// calibrate
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::calibrate(
    const std::shared_ptr<Matrix<dtype::real>> calibration_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (calibration_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::calibrate: calibration_voltage == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::calibrate: handle == nullptr");
    }

    // check size of voltage matrix
    if ((calibration_voltage->rows() != this->model()->source()->measurement_count()) ||
        (calibration_voltage->columns() != this->model()->source()->drive_count())) {
        throw std::invalid_argument("fastEIT::Solver::calibrate: calibration_voltage has incompatible shape");
    }

    // solve forward
    this->forward_solver()->solve(this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverse_solver()->calcSystemMatrix(this->model()->jacobian(), handle, stream);

    // solve inverse
    this->inverse_solver()->solve(this->model()->jacobian(), this->forward_solver()->voltage(),
        calibration_voltage, 180, handle, stream, this->dgamma());

    // add to gamma
    this->gamma()->add(this->dgamma(), stream);

    return this->gamma();
}
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::calibrate(
    cublasHandle_t handle, cudaStream_t stream) {
    // calibrate
    return this->calibrate(this->calibration_voltage(), handle, stream);
}

// solving
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::solve(
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
    const std::shared_ptr<Matrix<dtype::real>> calibration_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (measured_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::solve: measured_voltage == nullptr");
    }
    if (calibration_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::solve: calibration_voltage == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::solve: handle == nullptr");
    }

    // check size of voltage matrices
    if ((calibration_voltage->rows() != this->model()->source()->measurement_count()) ||
        (calibration_voltage->columns() != this->model()->source()->drive_count())) {
        throw std::invalid_argument("fastEIT::Solver::solve: calibration_voltage has incompatible shape");
    }
    if ((measured_voltage->rows() != this->model()->source()->measurement_count()) ||
        (measured_voltage->columns() != this->model()->source()->drive_count())) {
        throw std::invalid_argument("fastEIT::Solver::solve: measured_voltage has incompatible shape");
    }

    // solve
    this->inverse_solver()->solve(this->model()->jacobian(), calibration_voltage,
        measured_voltage, 180, handle, stream, this->dgamma());

    return this->dgamma();
}
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::solve(
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // solve
    return this->solve(measured_voltage, this->calibration_voltage(), handle, stream);
}
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::solve(
    cublasHandle_t handle, cudaStream_t stream) {
    // solve
    return this->solve(this->measured_voltage(), this->calibration_voltage(), handle, stream);
}
