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

// create inverse_solver
template <
    template <template <class> class> class numericalSolverType
>
mpFlow::EIT::InverseSolver<numericalSolverType>::InverseSolver(dtype::size elementCount,
    dtype::size measurementCount, dtype::index parallelImages,
    dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream)
    : regularizationFactor(regularizationFactor) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::InverseSolver: handle == nullptr");
    }

    // create matrices
    this->difference = std::make_shared<numeric::Matrix<dtype::real>>(measurementCount, parallelImages, stream);
    this->zeros = std::make_shared<numeric::Matrix<dtype::real>>(elementCount, parallelImages, stream);
    this->excitation = std::make_shared<numeric::Matrix<dtype::real>>(elementCount, parallelImages, stream);
    this->systemMatrix = std::make_shared<numeric::Matrix<dtype::real>>(elementCount, elementCount, stream);
    this->jacobianSquare = std::make_shared<numeric::Matrix<dtype::real>>(elementCount, elementCount, stream);

    // create numerical EIT
    this->numericalSolver = std::make_shared<numericalSolverType<mpFlow::numeric::Matrix>>(elementCount, parallelImages, stream);
}

// calc system matrix
template <
    template <template <class> class> class numericalSolverType
>
void mpFlow::EIT::InverseSolver<numericalSolverType>::calcSystemMatrix(
    const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::calcSystemMatrix: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::calcSystemMatrix: handle == nullptr");
    }

    // cublas coeficients
    dtype::real alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobianSquare->dataRows,
        this->jacobianSquare->dataCols, jacobian->dataRows, &alpha,
        jacobian->deviceData, jacobian->dataRows, jacobian->deviceData,
        jacobian->dataRows, &beta, this->jacobianSquare->deviceData,
        this->jacobianSquare->dataRows)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::EIT::InverseSolver::calcSystemMatrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    this->systemMatrix->copy(this->jacobianSquare, stream);

    // add lambda * Jt * J * Jt * J to systemMatrix
    beta = this->regularizationFactor;
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobianSquare->dataCols,
        this->jacobianSquare->dataRows, this->jacobianSquare->dataCols,
        &beta, this->jacobianSquare->deviceData,
        this->jacobianSquare->dataRows, this->jacobianSquare->deviceData,
        this->jacobianSquare->dataRows, &alpha, this->systemMatrix->deviceData,
        this->systemMatrix->dataRows) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error(
            "mpFlow::EIT::InverseSolver::calcSystemMatrix: add lambda * Jt * J * Jt * J to systemMatrix");
    }
}

// calc excitation
template <
    template <template <class> class> class numericalSolverType
>
void mpFlow::EIT::InverseSolver<numericalSolverType>::calcExcitation(
    const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::calcExcitation: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::calcExcitation: handle == nullptr");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // copy measuredVoltage to dVoltage
    for (dtype::index image = 0; image < this->numericalSolver->cols; ++image) {
        if (cublasScopy(handle, this->difference->dataRows,
            measurement[image]->deviceData, 1,
            (dtype::real*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::EIT::InverseSolver::calcExcitation: copy measuredVoltage to dVoltage");
        }

        // substract calculatedVoltage
        dtype::real alpha = -1.0f;
        if (cublasSaxpy(handle, this->difference->dataRows, &alpha,
            calculation[image]->deviceData, 1,
            (dtype::real*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::EIT::InverseSolver::calcExcitation: substract calculatedVoltage");
        }
    }

    // calc excitation
    dtype::real alpha = 1.0f;
    dtype::real beta = 0.0f;
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, jacobian->dataCols, this->difference->dataCols,
        jacobian->dataRows, &alpha, jacobian->deviceData, jacobian->dataRows, this->difference->deviceData,
        this->difference->dataRows, &beta, this->excitation->deviceData, this->excitation->dataRows)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::EIT::InverseSolver::calcExcitation: calc excitation");
    }
}

// inverse solving
template <
    template <template <class> class> class numericalSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> mpFlow::EIT::InverseSolver<numericalSolverType>::solve(
    const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
    const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream,
    std::shared_ptr<numeric::Matrix<dtype::real>> gamma) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::solve: jacobian == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::solve: handle == nullptr");
    }

    // reset gamma
    gamma->copy(this->zeros, stream);

    // calc excitation
    this->calcExcitation(jacobian, calculation, measurement, handle, stream);

    // solve system
    this->numericalSolver->solve(this->systemMatrix, this->excitation,
        steps, handle, stream, gamma);

    return gamma;
}

// specialisation
template class mpFlow::EIT::InverseSolver<mpFlow::numeric::ConjugateGradient>;
