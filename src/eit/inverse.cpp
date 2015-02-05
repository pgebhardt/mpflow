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
    class dataType,
    template <class, template <class> class> class numericalSolverType
>
mpFlow::EIT::InverseSolver<dataType, numericalSolverType>::InverseSolver(dtype::size elementCount,
    dtype::size measurementCount, dtype::index parallelImages,
    dataType regularizationFactor, cublasHandle_t handle, cudaStream_t stream)
    : regularizationFactor(regularizationFactor) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::InverseSolver: handle == nullptr");
    }

    // create matrices
    this->difference = std::make_shared<numeric::Matrix<dataType>>(measurementCount, parallelImages, stream, 0.0, false);
    this->excitation = std::make_shared<numeric::Matrix<dataType>>(elementCount, parallelImages, stream, 0.0, false);
    this->systemMatrix = std::make_shared<numeric::Matrix<dataType>>(elementCount, elementCount, stream, 0.0, false);
    this->jacobianSquare = std::make_shared<numeric::Matrix<dataType>>(elementCount, elementCount, stream, 0.0, false);

    // create numerical EIT
    this->numericalSolver = std::make_shared<numericalSolverType<dataType, mpFlow::numeric::Matrix>>(elementCount, parallelImages, stream);
}

// calc system matrix
template <
    class dataType,
    template <class, template <class> class> class numericalSolverType
>
void mpFlow::EIT::InverseSolver<dataType, numericalSolverType>::calcSystemMatrix(
    const std::shared_ptr<numeric::Matrix<dataType>> jacobian, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::calcSystemMatrix: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::InverseSolver::calcSystemMatrix: handle == nullptr");
    }

    // switch to correct stream
    cublasSetStream(handle, stream);

    // cublas coeficients
    dataType alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobianSquare->dataRows,
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
    if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobianSquare->dataCols,
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
    class dataType,
    template <class, template <class> class> class numericalSolverType
>
void mpFlow::EIT::InverseSolver<dataType, numericalSolverType>::calcExcitation(
    const std::shared_ptr<numeric::Matrix<dataType>> jacobian,
    const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& calculation,
    const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& measurement, cublasHandle_t handle,
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
        if (numeric::cublasWrapper<dataType>::copy(handle, this->difference->dataRows,
            measurement[image]->deviceData, 1,
            (dataType*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::EIT::InverseSolver::calcExcitation: copy measuredVoltage to dVoltage");
        }

        // substract calculatedVoltage
        dataType alpha = -1.0f;
        if (numeric::cublasWrapper<dataType>::axpy(handle, this->difference->dataRows, &alpha,
            calculation[image]->deviceData, 1,
            (dataType*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::EIT::InverseSolver::calcExcitation: substract calculatedVoltage");
        }
    }

    // calc excitation
    dataType alpha = 1.0f;
    dataType beta = 0.0f;
    if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, jacobian->dataCols, this->difference->dataCols,
        jacobian->dataRows, &alpha, jacobian->deviceData, jacobian->dataRows, this->difference->deviceData,
        this->difference->dataRows, &beta, this->excitation->deviceData, this->excitation->dataRows)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::EIT::InverseSolver::calcExcitation: calc excitation");
    }
}

// inverse solving
template <
    class dataType,
    template <class, template <class> class> class numericalSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>>
    mpFlow::EIT::InverseSolver<dataType, numericalSolverType>::solve(
    const std::shared_ptr<numeric::Matrix<dataType>> jacobian,
    const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& calculation,
    const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& measurement, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream,
    std::shared_ptr<numeric::Matrix<dataType>> gamma) {
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
    gamma->fill(0.0, stream);

    // calc excitation
    this->calcExcitation(jacobian, calculation, measurement, handle, stream);

    // solve system
    this->numericalSolver->solve(this->systemMatrix, this->excitation,
        steps, handle, stream, gamma);

    return gamma;
}

// specialisation
template class mpFlow::EIT::InverseSolver<mpFlow::dtype::real, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::EIT::InverseSolver<mpFlow::dtype::complex, mpFlow::numeric::ConjugateGradient>;
