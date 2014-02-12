// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "mpflow/mpflow.h"

// create inverse_solver
template <
    class numerical_solver
>
mpFlow::solver::Inverse<numerical_solver>::Inverse(dtype::size element_count,
    dtype::size voltage_count, dtype::index parallel_images,
    dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream)
    : regularization_factor_(regularization_factor) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::Inverse: handle == nullptr");
    }

    // create matrices
    this->difference_ = std::make_shared<numeric::Matrix<dtype::real>>(voltage_count, parallel_images, stream);
    this->zeros_ = std::make_shared<numeric::Matrix<dtype::real>>(element_count, parallel_images, stream);
    this->excitation_ = std::make_shared<numeric::Matrix<dtype::real>>(element_count, parallel_images, stream);
    this->system_matrix_ = std::make_shared<numeric::Matrix<dtype::real>>(element_count, element_count,
        stream);
    this->jacobian_square_ = std::make_shared<numeric::Matrix<dtype::real>>(element_count, element_count,
        stream);

    // create numeric solver
    this->numeric_solver_ = std::make_shared<numerical_solver>(element_count, parallel_images, stream);
}

// calc system matrix
template <
    class numerical_solver
>
void mpFlow::solver::Inverse<numerical_solver>::calcSystemMatrix(
    const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcSystemMatrix: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcSystemMatrix: handle == nullptr");
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
        throw std::logic_error("mpFlow::solver::Inverse::calcSystemMatrix: calc Jt * J");
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
            "mpFlow::solver::Inverse::calcSystemMatrix: add lambda * Jt * J * Jt * J to systemMatrix");
    }
}

// calc excitation
template <
    class numerical_solver
>
void mpFlow::solver::Inverse<numerical_solver>::calcExcitation(
    const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcExcitation: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcExcitation: handle == nullptr");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // copy measuredVoltage to dVoltage
    for (dtype::index image = 0; image < this->numeric_solver()->columns(); ++image) {
        if (cublasScopy(handle, this->difference()->data_rows(),
            measurement[image]->device_data(), 1,
            (dtype::real*)(this->difference()->device_data() + image * this->difference()->data_rows()), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::solver::Inverse::calcExcitation: copy measuredVoltage to dVoltage");
        }

        // substract calculatedVoltage
        dtype::real alpha = -1.0f;
        if (cublasSaxpy(handle, this->difference()->data_rows(), &alpha,
            calculation[image]->device_data(), 1,
            (dtype::real*)(this->difference()->device_data() + image * this->difference()->data_rows()), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::solver::Inverse::calcExcitation: substract calculatedVoltage");
        }
    }

    // calc excitation
    dtype::real alpha = 1.0f;
    dtype::real beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, jacobian->data_columns(), this->difference()->data_columns(),
        jacobian->data_rows(), &alpha, jacobian->device_data(), jacobian->data_rows(), this->difference()->device_data(),
        this->difference()->data_rows(), &beta, this->excitation()->device_data(), this->excitation()->data_rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::solver::Inverse::calcExcitation: calc excitation");
    }
}

// inverse solving
template <
    class numerical_solver
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> mpFlow::solver::Inverse<numerical_solver>::solve(
    const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream,
    std::shared_ptr<numeric::Matrix<dtype::real>> gamma) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::solve: jacobian == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::solve: handle == nullptr");
    }

    // reset gamma
    gamma->copy(this->zeros(), stream);

    // calc excitation
    this->calcExcitation(jacobian, calculation, measurement, handle, stream);

    // solve system
    this->numeric_solver()->solve(this->system_matrix(), this->excitation(),
        steps, handle, stream, gamma);

    return gamma;
}

// specialisation
template class mpFlow::solver::Inverse<mpFlow::numeric::Conjugate>;
template class mpFlow::solver::Inverse<mpFlow::numeric::FastConjugate>;
