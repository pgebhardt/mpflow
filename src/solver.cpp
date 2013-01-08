// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

// create solver
template <
    class model_type,
    class source_type
>
fastEIT::Solver<model_type, source_type>::Solver(std::shared_ptr<model_type> model,
    std::shared_ptr<source_type> source, dtype::real regularization_factor,
    cublasHandle_t handle, cudaStream_t stream)
    : model_(model), source_(source) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("Solver::Solver: model == nullptr");
    }
    if (source == nullptr) {
        throw std::invalid_argument("Solver::Solver: source == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Solver::Solver: handle == NULL");
    }

    // create forward solver
    this->forward_solver_ = std::make_shared<ForwardSolver<numeric::SparseConjugate,
        model_type, source_type>>(this->model(), source, handle, stream);

    // create inverse solver
    this->inverse_solver_ = std::make_shared<InverseSolver<numeric::Conjugate>>(
        this->model()->mesh()->elements()->rows(),
        this->source()->measurement_pattern()->data_columns() *
        this->source()->drive_pattern()->data_columns(),
        regularization_factor, handle, stream);

    // create matrices
    this->dgamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), 1, stream);
    this->gamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), 1, stream);
    this->measured_voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->source()->measurement_count(),
        this->source()->drive_count(), stream);
    this->calibration_voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->source()->measurement_count(),
        this->source()->drive_count(), stream);
}

// pre solve for accurate initial jacobian
template <
    class model_type,
    class source_type
>
void fastEIT::Solver<model_type, source_type>::preSolve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Solver::pre_solve: handle == NULL");
    }

    // forward solving a few steps
    this->forward_solver()->solve(this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    this->measured_voltage()->copy(this->forward_solver()->voltage(), stream);
    this->calibration_voltage()->copy(this->forward_solver()->voltage(), stream);
}

// calibrate
template <
    class model_type,
    class source_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<model_type, source_type>::calibrate(
    const std::shared_ptr<Matrix<dtype::real>> calibration_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (calibration_voltage == nullptr) {
        throw std::invalid_argument("Solver::calibrate: calibration_voltage == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Solver::calibrate: handle == NULL");
    }

    // solve forward
    this->forward_solver()->solve(this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->jacobian(), handle, stream);

    // solve inverse
    this->inverse_solver()->solve(this->forward_solver()->jacobian(), this->forward_solver()->voltage(),
        calibration_voltage, 90, true, handle, stream, this->dgamma());

    // add to gamma
    this->gamma()->add(this->dgamma(), stream);

    return this->gamma();
}
template <
    class model_type,
    class source_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<model_type, source_type>::calibrate(
    cublasHandle_t handle, cudaStream_t stream) {
    // calibrate
    return this->calibrate(this->calibration_voltage(), handle, stream);
}

// solving
template <
    class model_type,
    class source_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<model_type, source_type>::solve(
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
    const std::shared_ptr<Matrix<dtype::real>> calibration_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (measured_voltage == nullptr) {
        throw std::invalid_argument("Solver::solve: measured_voltage == nullptr");
    }
    if (calibration_voltage == nullptr) {
        throw std::invalid_argument("Solver::solve: calibration_voltage == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Solver::solve: handle == NULL");
    }

    // solve
    this->inverse_solver()->solve(this->forward_solver()->jacobian(), calibration_voltage,
        measured_voltage, 90, false, handle, stream, this->dgamma());

    return this->dgamma();
}
template <
    class model_type,
    class source_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<model_type, source_type>::solve(
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // solve
    return this->solve(measured_voltage, this->calibration_voltage(), handle, stream);
}
template <
    class model_type,
    class source_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<model_type, source_type>::solve(
    cublasHandle_t handle, cudaStream_t stream) {
    // solve
    return this->solve(this->measured_voltage(), this->calibration_voltage(), handle, stream);
}

// specialization
template class fastEIT::Solver<fastEIT::Model<fastEIT::basis::Linear>, fastEIT::source::Current>;
template class fastEIT::Solver<fastEIT::Model<fastEIT::basis::Linear>, fastEIT::source::Voltage>;
