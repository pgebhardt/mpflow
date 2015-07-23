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
    : regularizationFactor_(dataType(0)), regularizationType_(Inverse<dataType, numericalSolverType>::identity),
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
    this->result = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), parallelImages, stream, 0.0, false);
        
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
    this->systemMatrix->copy(this->regularizationMatrix, stream);
    this->systemMatrix->scalarMultiply(this->regularizationFactor(), stream);
    this->systemMatrix->add(this->jacobianSquare, stream);
}

template <
    class dataType,
    template <class> class numericalSolverType
>
void mpFlow::solver::Inverse<dataType, numericalSolverType>::calcRegularizationMatrix(
    cublasHandle_t const handle, cudaStream_t const stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::calcRegularizationMatrix: handle == nullptr");
    }

    // calculate regularization matrix according to type
    if (this->regularizationType() == RegularizationType::identity) {
        this->regularizationMatrix->setEye(stream);
    }
    else if (this->regularizationType() == RegularizationType::diagonal) {
        this->regularizationMatrix->diag(this->jacobianSquare, stream);
    }
    else if (this->regularizationType() == RegularizationType::totalVariational) {
        // calculate connection matrix
        Eigen::ArrayXf signs = Eigen::ArrayXf::Ones(this->mesh->edges.size());
        auto L = std::make_shared<numeric::Matrix<dataType>>(this->mesh->edges.rows(),
            this->mesh->elements.rows(), stream);
        for (int element = 0; element < this->mesh->elements.rows(); ++element) {
            for (unsigned i = 0; i < this->mesh->elementEdges.cols(); ++i) {
                auto const edge = this->mesh->elementEdges(element, i);
                
                auto const length = std::sqrt(
                    math::square(this->mesh->nodes(this->mesh->edges(edge, 0), 0) -
                        this->mesh->nodes(this->mesh->edges(edge, 1), 0)) +
                    math::square(this->mesh->nodes(this->mesh->edges(edge, 0), 1) -
                        this->mesh->nodes(this->mesh->edges(edge, 1), 1)));
                        
                (*L)(edge, element) = signs(edge) * length;
                signs(this->mesh->elementEdges(element, i)) *= -1;
            }
        }
        L->copyToDevice(stream);
        cudaStreamSynchronize(stream);
        
        // calculate regularization matrix
        this->regularizationMatrix->multiply(L, L, handle, stream, CUBLAS_OP_T);
    }
    else {
        throw std::runtime_error(
            "mpFlow::solver::Inverse::calcRegularizationMatrix: regularization type not implemented yet");
    }
    
    // update system matrix
    this->systemMatrix->copy(this->regularizationMatrix, stream);
    this->systemMatrix->scalarMultiply(this->regularizationFactor(), stream);
    this->systemMatrix->add(this->jacobianSquare, stream);
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

    // calc Jt * J
    dataType const alpha = 1.0, beta = 0.0;
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
        dataType const alpha = -1.0;
        if (numeric::cublasWrapper<dataType>::axpy(handle, this->difference->dataRows, &alpha,
            calculation[image]->deviceData, 1,
            (dataType*)(this->difference->deviceData + image * this->difference->dataRows), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "mpFlow::solver::Inverse::calcExcitation: substract calculatedVoltage");
        }
    }

    // calc excitation
    dataType const alpha = 1.0, beta = 0.0;
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
std::shared_ptr<mpFlow::numeric::Matrix<dataType> const>
    mpFlow::solver::Inverse<dataType, numericalSolverType>::solve(
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
    std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
    cublasHandle_t const handle, cudaStream_t const stream, unsigned const maxIterations,
    unsigned* const iterations) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::solver::Inverse::solve: handle == nullptr");
    }

    // reset result vector
    this->result->fill(dataType(0), stream);

    // calc excitation
    this->calcExcitation(calculation, measurement, handle, stream);

    // solve system
    unsigned const steps = this->numericalSolver->template solve<numeric::Matrix, numeric::Matrix>(
        this->systemMatrix, this->excitation, handle, stream, this->result, nullptr, maxIterations);
        
    if (iterations != nullptr) {
        *iterations = steps; 
    }
    
    return this->result;
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
