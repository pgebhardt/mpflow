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
    template <class> class numericalSolverType
>
mpFlow::solver::Inverse<dataType, numericalSolverType>::Inverse(unsigned const elementCount,
    unsigned const measurementCount, unsigned const parallelImages,
    dataType const regularizationFactor, cublasHandle_t const handle,
    cudaStream_t const stream)
    : regularizationFactor(regularizationFactor) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::Inverse: handle == nullptr");
    }

    // create matrices
    this->difference = std::make_shared<numeric::Matrix<dataType>>(measurementCount, parallelImages, stream, 0.0, false);
    this->excitation = std::make_shared<numeric::Matrix<dataType>>(elementCount, parallelImages, stream, 0.0, false);
    this->systemMatrix = std::make_shared<numeric::Matrix<dataType>>(elementCount, elementCount, stream, 0.0, false);
    this->jacobianSquare = std::make_shared<numeric::Matrix<dataType>>(elementCount, elementCount, stream, 0.0, false);

    // create numerical EIT
    this->numericalSolver = std::make_shared<numericalSolverType<dataType>>(elementCount, parallelImages, stream);
}

// calc system matrix
template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::calcSystemMatrix(
    std::shared_ptr<numeric::Matrix<dataType> const> jacobian, RegularizationType const regularizationType,
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcSystemMatrix: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcSystemMatrix: handle == nullptr");
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
        throw std::logic_error("mpFlow::solver::Inverse::calcSystemMatrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    this->systemMatrix->copy(this->jacobianSquare, stream);

    // choose regularizationType
    if (regularizationType == RegularizationType::square) {
        // add lambda * Jt * J * Jt * J to systemMatrix
        beta = this->regularizationFactor;
        if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobianSquare->dataCols,
            this->jacobianSquare->dataRows, this->jacobianSquare->dataCols,
            &beta, this->jacobianSquare->deviceData,
            this->jacobianSquare->dataRows, this->jacobianSquare->deviceData,
            this->jacobianSquare->dataRows, &alpha, this->systemMatrix->deviceData,
            this->systemMatrix->dataRows) != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::solver::Inverse::calcSystemMatrix: add lambda * Jt * J * Jt * J to systemMatrix");
        }
    }
    else {
        auto diag = numeric::Matrix<dataType>::eye(this->systemMatrix->rows, stream);
        diag->scalarMultiply(this->regularizationFactor, stream);
        this->systemMatrix->add(diag, stream);
    }
}

// calc excitation
template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::calcExcitation(
    std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
    cublasHandle_t const handle, cudaStream_t const stream) {
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
    for (unsigned image = 0; image < this->numericalSolver->cols; ++image) {
        if (numeric::cublasWrapper<dataType>::copy(handle, this->difference->dataRows,
            measurement[image]->deviceData, 1,
            (dataType*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::solver::Inverse::calcExcitation: copy measuredVoltage to dVoltage");
        }

        // substract calculatedVoltage
        dataType alpha = -1.0f;
        if (numeric::cublasWrapper<dataType>::axpy(handle, this->difference->dataRows, &alpha,
            calculation[image]->deviceData, 1,
            (dataType*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::solver::Inverse::calcExcitation: substract calculatedVoltage");
        }
    }

    // calc excitation
    dataType alpha = 1.0f;
    dataType beta = 0.0f;
    if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, jacobian->dataCols, this->difference->dataCols,
        jacobian->dataRows, &alpha, jacobian->deviceData, jacobian->dataRows, this->difference->deviceData,
        this->difference->dataRows, &beta, this->excitation->deviceData, this->excitation->dataRows)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::solver::Inverse::calcExcitation: calc excitation");
    }
}

// inverse solving
template <
    class dataType,
    template <class> class numericalSolverType
>
unsigned mpFlow::solver::Inverse<dataType, numericalSolverType>::solve(
    std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
    unsigned const steps, cublasHandle_t const handle, cudaStream_t const stream,
    std::shared_ptr<numeric::Matrix<dataType>> gamma) {
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
    gamma->fill(0.0, stream);

    // calc excitation
    this->calcExcitation(jacobian, calculation, measurement, handle, stream);

    // solve system
    return this->numericalSolver->template solve<numeric::Matrix>(this->systemMatrix,
        this->excitation, handle, stream, gamma, nullptr, steps);
}

// specialisation
template class mpFlow::solver::Inverse<float, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Inverse<double, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Inverse<thrust::complex<float>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Inverse<thrust::complex<double>, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::solver::Inverse<float, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Inverse<double, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Inverse<thrust::complex<float>, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::solver::Inverse<thrust::complex<double>, mpFlow::numeric::BiCGSTAB>;
