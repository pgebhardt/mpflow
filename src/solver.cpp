// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"

// create solver
fastEIT::Solver::Solver(std::shared_ptr<fastEIT::model::Model> model, dtype::index parallel_images,
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
        parallel_images, regularization_factor, handle, stream);

    // create matrices
    this->dgamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), parallel_images, stream);
    this->gamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), parallel_images, stream);
    for (dtype::index image = 0; image < parallel_images; ++image) {
        this->measured_voltage_.push_back(std::make_shared<Matrix<dtype::real>>(
            this->model()->source()->measurement_count(),
            this->model()->source()->drive_count(), stream));
        this->calibration_voltage_.push_back(std::make_shared<Matrix<dtype::real>>(
            this->model()->source()->measurement_count(),
            this->model()->source()->drive_count(), stream));
    }
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
    for (auto voltage : this->measured_voltage_) {
        voltage->copy(this->forward_solver()->voltage(), stream);
    }
    for (auto voltage : this->calibration_voltage_) {
        voltage->copy(this->forward_solver()->voltage(), stream);
    }
}
/*
// calibrate
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::calibrate(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::calibrate: handle == nullptr");
    }

    // solve forward
    this->forward_solver()->solve(this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverse_solver()->calcSystemMatrix(this->model()->jacobian(), handle, stream);

    // solve inverse
    this->inverse_solver()->solve(this->model()->jacobian(), this->forward_solver()->voltage(),
        this->calibration_voltage_, 180, handle, stream, this->dgamma());

    // add to gamma
    this->gamma()->add(this->dgamma(), stream);

    return this->gamma();
}
*/
// solving
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver::solve(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Solver::solve: handle == nullptr");
    }

    // solve
    this->inverse_solver()->solve(this->model()->jacobian(), this->calibration_voltage_,
        this->measured_voltage_, 180, handle, stream, this->dgamma());

    return this->dgamma();
}
