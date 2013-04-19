// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"

// create inverse_solver
template <
    class NumericSolver
>
fastEIT::InverseSolver<NumericSolver>::InverseSolver(dtype::size element_count,
    dtype::size voltage_count, dtype::real regularization_factor,
    cublasHandle_t handle, cudaStream_t stream)
    : regularization_factor_(regularization_factor) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::InverseSolver: handle == nullptr");
    }

    // create matrices
    this->dvoltage_ = std::make_shared<Matrix<dtype::real>>(voltage_count, 1, stream);
    this->zeros_ = std::make_shared<Matrix<dtype::real>>(element_count, 1, stream);
    this->excitation_ = std::make_shared<Matrix<dtype::real>>(element_count, 1, stream);
    this->system_matrix_ = std::make_shared<Matrix<dtype::real>>(element_count, element_count,
        stream);
    this->jacobian_square_ = std::make_shared<Matrix<dtype::real>>(element_count, element_count,
        stream);

    // create numeric solver
    this->numeric_solver_ = std::make_shared<NumericSolver>(element_count, handle, stream);
}

// calc system matrix
template <
    class NumericSolver
>
void fastEIT::InverseSolver<NumericSolver>::calcSystemMatrix(
    const std::shared_ptr<Matrix<dtype::real>> jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::calcSystemMatrix: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::calcSystemMatrix: handle == nullptr");
    }

    // cublas coeficients
    dtype::real alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobian_square()->data_rows(),
        this->jacobian_square()->data_columns(), jacobian->data_rows(), &alpha,
        jacobian->device_data(), jacobian->data_rows(), jacobian->device_data(),
        jacobian->data_rows(), &beta, this->jacobian_square()->device_data(),
        this->jacobian_square()->data_rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("fastEIT::InverseSolver::calcSystemMatrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    this->system_matrix()->copy(this->jacobian_square(), stream);

    // add lambda * Jt * J * Jt * J to systemMatrix
    beta = this->regularization_factor();
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobian_square()->data_columns(),
        this->jacobian_square()->data_rows(), this->jacobian_square()->data_columns(),
        &beta, this->jacobian_square()->device_data(),
        this->jacobian_square()->data_rows(), this->jacobian_square()->device_data(),
        this->jacobian_square()->data_rows(), &alpha, this->system_matrix()->device_data(),
        this->system_matrix()->data_rows()) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "fastEIT::InverseSolver::calcSystemMatrix: add lambda * Jt * J * Jt * J to systemMatrix");
    }
}

// calc excitation
template <
    class NumericSolver
>
void fastEIT::InverseSolver<NumericSolver>::calcExcitation(
    const std::shared_ptr<Matrix<dtype::real>> jacobian,
    const std::shared_ptr<Matrix<dtype::real>> calculated_voltage,
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::calcExcitation: jacobian == nullptr");
    }
    if (calculated_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::calcExcitation: calculated_voltage == nullptr");
    }
    if (measured_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::calcExcitation: measured_voltage == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::calcExcitation: handle == nullptr");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // copy measuredVoltage to dVoltage
    if (cublasScopy(handle, this->dvoltage()->data_rows(),
        measured_voltage->device_data(), 1, this->dvoltage()->device_data(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "fastEIT::InverseSolver::calcExcitation: copy measuredVoltage to dVoltage");
    }

    // substract calculatedVoltage
    dtype::real alpha = -1.0f;
    if (cublasSaxpy(handle, this->dvoltage()->data_rows(), &alpha,
        calculated_voltage->device_data(), 1, this->dvoltage()->device_data(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "fastEIT::InverseSolver::calcExcitation: substract calculatedVoltage");
    }

    // calc excitation
    alpha = 1.0f;
    dtype::real beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->data_rows(), jacobian->data_columns(), &alpha,
        jacobian->device_data(), jacobian->data_rows(), this->dvoltage()->device_data(), 1, &beta,
        this->excitation()->device_data(), 1) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("fastEIT::InverseSolver::calcExcitation: calc excitation");
    }
}

// inverse solving
template <
    class NumericSolver
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::InverseSolver<NumericSolver>::solve(
    const std::shared_ptr<Matrix<dtype::real>> jacobian,
    const std::shared_ptr<Matrix<dtype::real>> calculated_voltage,
    const std::shared_ptr<Matrix<dtype::real>> measured_voltage, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream,
    std::shared_ptr<Matrix<dtype::real>> gamma) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::solve: jacobian == nullptr");
    }
    if (calculated_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::solve: calculated_voltage == nullptr");
    }
    if (measured_voltage == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::solve: measured_voltage == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::InverseSolver::solve: handle == nullptr");
    }

    // reset gamma
    gamma->copy(this->zeros(), stream);

    // calc excitation
    this->calcExcitation(jacobian, calculated_voltage, measured_voltage, handle, stream);

    // solve system
    this->numeric_solver()->solve(this->system_matrix(), this->excitation(),
        steps, handle, stream, gamma);

    return gamma;
}

// specialisation
template class fastEIT::InverseSolver<fastEIT::numeric::Conjugate>;
