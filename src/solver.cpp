// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <assert.h>

#include <stdexcept>
#include <vector>
#include <array>
#include <tuple>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/matrix.h"
#include "../include/sparse_matrix.h"
#include "../include/basis.h"
#include "../include/mesh.h"
#include "../include/electrodes.h"
#include "../include/conjugate.h"
#include "../include/sparse_conjugate.h"
#include "../include/model.h"
#include "../include/forward.h"
#include "../include/inverse.h"
#include "../include/solver.h"

// create solver
template <
    class BasisFunction
>
fastEIT::Solver<BasisFunction>::Solver(std::shared_ptr<Model<BasisFunction>> model,
    const std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
    const std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream)
    : model_(model) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("Solver::Solver: model == nullptr");
    }
    if (measurement_pattern == nullptr) {
        throw std::invalid_argument("Solver::Solver: measurement_pattern == nullptr");
    }
    if (drive_pattern == nullptr) {
        throw std::invalid_argument("Solver::Solver: drive_pattern == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Solver::Solver: handle == NULL");
    }

    // create forward solver
    this->forward_solver_ = std::make_shared<ForwardSolver<BasisFunction, numeric::SparseConjugate>>(
        this->model(), measurement_pattern, drive_pattern, handle, stream);

    // create inverse solver
    this->inverse_solver_ = std::make_shared<InverseSolver<numeric::Conjugate>>(
        this->model()->mesh()->elements()->rows(),
        measurement_pattern->data_columns() * drive_pattern->data_columns(),
        regularization_factor, handle, stream);

    // create matrices
    this->dgamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), 1, stream);
    this->gamma_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->elements()->rows(), 1, stream);
    this->measured_voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->forward_solver()->measurement_count(),
        this->forward_solver()->drive_count(), stream);
    this->calibration_voltage_ = std::make_shared<Matrix<dtype::real>>(
        this->forward_solver()->measurement_count(),
        this->forward_solver()->drive_count(), stream);
}

// pre solve for accurate initial jacobian
template <
    class BasisFunction
>
void fastEIT::Solver<BasisFunction>::preSolve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Solver::pre_solve: handle == NULL");
    }

    // forward solving a few steps
    this->forward_solver()->solve(this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverse_solver()->calcSystemMatrix(this->forward_solver()->jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    this->measured_voltage()->copy(this->forward_solver()->voltage().get(), stream);
    this->calibration_voltage()->copy(this->forward_solver()->voltage().get(), stream);
}

// calibrate
template <
    class BasisFunction
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<BasisFunction>::calibrate(
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
    this->gamma()->add(this->dgamma().get(), stream);

    return this->gamma();
}
template <
    class BasisFunction
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<BasisFunction>::calibrate(
    cublasHandle_t handle, cudaStream_t stream) {
    // calibrate
    return this->calibrate(this->calibration_voltage(), handle, stream);
}

// solving
template <
    class BasisFunction
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<BasisFunction>::solve(
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
    class BasisFunction
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<BasisFunction>::solve(
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // solve
    return this->solve(measured_voltage, this->calibration_voltage(), handle, stream);
}
template <
    class BasisFunction
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Solver<BasisFunction>::solve(
    cublasHandle_t handle, cudaStream_t stream) {
    // solve
    return this->solve(this->measured_voltage(), this->calibration_voltage(), handle, stream);
}

// specialization
template class fastEIT::Solver<fastEIT::basis::Linear>;
