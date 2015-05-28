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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"

// create inverse_solver
template <
    class dataType,
    template <class> class numericalSolverType
>
mpFlow::solver::Inverse<dataType, numericalSolverType>::Inverse(
    std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
    unsigned const parallelImages, cublasHandle_t const handle, cudaStream_t const stream)
    : regularizationFactor_(dataType(0)), regularizationType_(Inverse<dataType, numericalSolverType>::unity),
    jacobian(jacobian), mesh(mesh) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::Inverse: handle == nullptr");
    }

    // create matrices
    this->difference = std::make_shared<numeric::Matrix<dataType>>(jacobian->rows, parallelImages, stream, 0.0, false);
    this->excitation = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), parallelImages, stream, 0.0, false);
    this->jacobianSquare = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), mesh->elements.rows(), stream, 0.0, false);
    this->regularizationMatrix = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), mesh->elements.rows(), stream, 0.0, false);
    this->systemMatrix = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), mesh->elements.rows(), stream, 0.0, false);
    
    // create numerical EIT
    this->numericalSolver = std::make_shared<numericalSolverType<dataType>>(mesh->elements.rows(), parallelImages, stream);
        
    // initialize matrices
    this->updateJacobian(jacobian, handle, stream);
}

template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::updateJacobian(
    std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::updateJacobian: jacobian == nullptr");
    }

    // save jacobian for future use
    this->jacobian = jacobian;
    
    // update system matrix
    this->calcJacobianSquare(handle, stream);
    this->systemMatrix->copy(this->jacobianSquare, stream);
    this->systemMatrix->add(this->regularizationMatrix, stream);
}

template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::calcRegularizationMatrix(
    cudaStream_t const stream) {
    // calculate regularization matrix according to type
    if (this->regularizationType() == RegularizationType::unity) {
        this->regularizationMatrix->setEye(stream);
    }
    else if (this->regularizationType() == RegularizationType::diagonal) {
        this->regularizationMatrix->diag(this->jacobianSquare, stream);
    }
    else {
        throw std::runtime_error(
            "mpFlow::solver::Inverse::calcRegularizationMatrix: regularization type not implemented yet");
    }
    
    // apply regularization factor
    this->regularizationMatrix->scalarMultiply(this->regularizationFactor(), stream);
    
    // update system matrix
    this->systemMatrix->copy(this->jacobianSquare, stream);
    this->systemMatrix->add(this->regularizationMatrix, stream);
}

template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::calcJacobianSquare(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcJacobianSquare: handle == nullptr");
    }

    // switch to correct stream
    cublasSetStream(handle, stream);

    // cublas coeficients
    dataType alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobianSquare->dataRows,
        this->jacobianSquare->dataCols, this->jacobian->dataRows, &alpha,
        this->jacobian->deviceData, this->jacobian->dataRows, this->jacobian->deviceData,
        this->jacobian->dataRows, &beta, this->jacobianSquare->deviceData,
        this->jacobianSquare->dataRows)
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::solver::Inverse::calcJacobianSquare: calc Jt * J");
    }
}

// calc excitation
template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::calcExcitation(
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
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
    if (numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobian->dataCols,
        this->difference->dataCols, this->jacobian->dataRows, &alpha, this->jacobian->deviceData,
        this->jacobian->dataRows, this->difference->deviceData, this->difference->dataRows, &beta,
        this->excitation->deviceData, this->excitation->dataRows)
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
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
    unsigned const steps, cublasHandle_t const handle, cudaStream_t const stream,
    std::shared_ptr<numeric::Matrix<dataType>> result) {
    // check input
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::solve: result == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::solve: handle == nullptr");
    }

    // reset result vector
    result->fill(dataType(0), stream);

    // calc excitation
    this->calcExcitation(calculation, measurement, handle, stream);

    // solve system
    return this->numericalSolver->template solve<numeric::Matrix, numeric::Matrix>(
        this->systemMatrix, this->excitation, handle, stream, result, nullptr, steps);
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
