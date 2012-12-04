// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <vector>
#include <array>
#include <tuple>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/matrix.hpp"
#include "../include/mesh.hpp"
#include "../include/electrodes.hpp"
#include "../include/conjugate.hpp"
#include "../include/inverse.hpp"

// create inverse_solver
template <
    class NumericSolver
>
fastEIT::InverseSolver<NumericSolver>::InverseSolver(dtype::size element_count,
    dtype::size voltage_count, dtype::real regularization_factor,
    cublasHandle_t handle, cudaStream_t stream)
    : numeric_solver_(NULL), dvoltage_(NULL), zeros_(NULL), excitation_(NULL),
        system_matrix_(NULL), jacobian_square_(NULL),
        regularization_factor_(regularization_factor) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("InverseSolver::InverseSolver: handle == NULL");
    }

    // create matrices
    this->dvoltage_ = new Matrix<dtype::real>(voltage_count, 1, stream);
    this->zeros_ = new Matrix<dtype::real>(element_count, 1, stream);
    this->excitation_ = new Matrix<dtype::real>(element_count, 1, stream);
    this->system_matrix_ = new Matrix<dtype::real>(element_count, element_count, stream);
    this->jacobian_square_ = new Matrix<dtype::real>(element_count, element_count, stream);

    // create numeric solver
    this->numeric_solver_ = new NumericSolver(element_count, handle, stream);
}

// release solver
template <
    class NumericSolver
>
fastEIT::InverseSolver<NumericSolver>::~InverseSolver() {
    // cleanup
    delete this->numeric_solver_;
    delete this->dvoltage_;
    delete this->zeros_;
    delete this->excitation_;
    delete this->system_matrix_;
    delete this->jacobian_square_;
}

// calc system matrix
template <
    class NumericSolver
>
void fastEIT::InverseSolver<NumericSolver>::calcSystemMatrix(
    const Matrix<dtype::real>& jacobian, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("InverseSolver::calcSystemMatrix: handle == NULL");
    }

    // cublas coeficients
    dtype::real alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobian_square().data_rows(),
        this->jacobian_square().data_columns(), jacobian.data_rows(), &alpha,
        jacobian.device_data(), jacobian.data_rows(), jacobian.device_data(),
        jacobian.data_rows(), &beta, this->jacobian_square().device_data(),
        this->jacobian_square().data_rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("InverseSolver::calcSystemMatrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    this->system_matrix().copy(this->jacobian_square(), stream);

    // add lambda * Jt * J * Jt * J to systemMatrix
    beta = this->regularization_factor();
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobian_square().data_columns(),
        this->jacobian_square().data_rows(), this->jacobian_square().data_columns(),
        &beta, this->jacobian_square().device_data(),
        this->jacobian_square().data_rows(), this->jacobian_square().device_data(),
        this->jacobian_square().data_rows(), &alpha, this->system_matrix().device_data(),
        this->system_matrix().data_rows()) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "InverseSolver::calcSystemMatrix: add lambda * Jt * J * Jt * J to systemMatrix");
    }
}

// calc excitation
template <
    class NumericSolver
>
void fastEIT::InverseSolver<NumericSolver>::calcExcitation(
    const Matrix<dtype::real>& jacobian, const Matrix<dtype::real>& calculated_voltage,
    const Matrix<dtype::real>& measured_voltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("InverseSolver::calcExcitation: handle == NULL");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // copy measuredVoltage to dVoltage
    if (cublasScopy(handle, this->dvoltage().data_rows(),
        calculated_voltage.device_data(), 1, this->dvoltage().device_data(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "InverseSolver::calcExcitation: copy measuredVoltage to dVoltage");
    }

    // substract calculatedVoltage
    dtype::real alpha = -1.0f;
    if (cublasSaxpy(handle, this->dvoltage().data_rows(), &alpha,
        measured_voltage.device_data(), 1, this->dvoltage().device_data(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "Model::calcExcitation: substract calculatedVoltage");
    }

    // calc excitation
    alpha = 1.0f;
    dtype::real beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian.data_rows(), jacobian.data_columns(), &alpha,
        jacobian.device_data(), jacobian.data_rows(), this->dvoltage().device_data(), 1, &beta,
        this->excitation().device_data(), 1) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("InverseSolver::calcExcitation: calc excitation");
    }
}

// inverse solving
template <
    class NumericSolver
>
const fastEIT::Matrix<fastEIT::dtype::real>& fastEIT::InverseSolver<NumericSolver>::solve(
    const Matrix<dtype::real>& jacobian, const Matrix<dtype::real>& calculated_voltage,
    const Matrix<dtype::real>& measured_voltage, dtype::size steps, bool regularized,
    cublasHandle_t handle, cudaStream_t stream, Matrix<dtype::real>* gamma) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("InverseSolver::solve: handle == NULL");
    }
    if (gamma == NULL) {
        throw std::invalid_argument("InverseSolver::solve: gamma == NULL");
    }

    // reset gamma
    gamma->copy(this->zeros(), stream);

    // calc excitation
    this->calcExcitation(jacobian, calculated_voltage, measured_voltage, handle, stream);

    // solve system
    this->numeric_solver().solve(regularized ? this->system_matrix() : this->jacobian_square(),
        this->excitation(), steps, handle, stream, gamma);

    return *gamma;
}

// specialisation
template class fastEIT::InverseSolver<fastEIT::numeric::Conjugate>;
