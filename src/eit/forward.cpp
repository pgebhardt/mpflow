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
#include "mpflow/eit/forward_kernel.h"

// create forward_solver
template <
    template <class> class numericalSolverType,
    class equationType
>
mpFlow::EIT::ForwardSolver<numericalSolverType, equationType>::ForwardSolver(
    std::shared_ptr<equationType> const equation,
    std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
    unsigned const components, cublasHandle_t const handle, cudaStream_t const stream)
    : equation(equation), source(source) {
    // check input
    if (equation == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::ForwardSolver::ForwardSolver: equation == nullptr");
    }
    if (components < 1) {
        throw std::invalid_argument(
            "mpFlow::EIT::ForwardSolver::ForwardSolver: components < 1");
    }
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::ForwardSolver::ForwardSolver: handle == nullptr");
    }

    // create numericalSolver solver
    this->numericalSolver = std::make_shared<numericalSolverType<dataType>>(
        this->equation->mesh->nodes.rows(),
        this->source->drivePattern->cols + this->source->measurementPattern->cols, stream);

    // create matrices
    this->result = std::make_shared<numeric::Matrix<dataType>>(
        this->source->measurementPattern->cols, this->source->drivePattern->cols, stream);
    for (unsigned component = 0; component < components; ++component) {
        this->phi.push_back(std::make_shared<numeric::Matrix<dataType>>(this->equation->mesh->nodes.rows(),
            this->source->pattern->cols, stream));
    }
    this->excitation = std::make_shared<numeric::Matrix<dataType>>(this->equation->mesh->nodes.rows(),
        this->source->pattern->cols, stream);
    this->jacobian = std::make_shared<numeric::Matrix<dataType>>(
        this->source->measurementPattern->dataCols * this->source->drivePattern->dataCols,
        this->equation->mesh->elements.rows(), stream, 0.0, false);
    this->preconditioner = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->equation->systemMatrix->rows, this->equation->systemMatrix->cols, stream);

    // create matrix to calculate system excitation from electrode excitation
    this->electrodesAttachmentMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->source->measurementPattern->cols,
        this->equation->mesh->nodes.rows(), stream, 0.0, false);
          
    if (this->source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
        applyMixedBoundaryCondition(this->equation->excitationMatrix, this->equation->systemMatrix, stream);
    }
    
    this->electrodesAttachmentMatrix->multiply(this->source->measurementPattern,
        this->equation->excitationMatrix, handle, stream, CUBLAS_OP_T, CUBLAS_OP_T);
}

// forward solving
template <
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::EIT::ForwardSolver<numericalSolverType, equationType>::solve(
    std::shared_ptr<numeric::Matrix<dataType> const> const gamma, cublasHandle_t const handle,
    cudaStream_t const stream, unsigned* const steps) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: handle == nullptr");
    }

    // calculate common excitation for all 2.5D model components
    unsigned totalSteps = 0;
    for (unsigned component = 0; component < this->phi.size(); ++component) {
        // 2.5D model constants
        dataType alpha = math::square(2.0 * component * M_PI / this->equation->mesh->height);
        dataType beta = component == 0 ? (1.0 / this->equation->mesh->height) :
            (2.0 * sin(component * M_PI * this->equation->boundaryDescriptor->height / this->equation->mesh->height) /
                (component * M_PI * this->equation->boundaryDescriptor->height));

        // update system matrix for different 2.5D components
        this->equation->update(gamma, alpha, gamma, stream);

        if (this->source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
            applyMixedBoundaryCondition(this->equation->excitationMatrix, this->equation->systemMatrix, stream);
        }

        this->excitation->multiply(this->equation->excitationMatrix,
            this->source->pattern, handle, stream);

        if (this->source->type == FEM::SourceDescriptor<dataType>::Type::Open) {
            this->excitation->scalarMultiply(beta, stream);
        }

        // solve linear system
        numeric::preconditioner::diagonal<dataType>(this->equation->systemMatrix, stream, this->preconditioner);
        totalSteps += this->numericalSolver->solve(this->equation->systemMatrix,
            this->excitation, nullptr, stream, this->phi[component], this->preconditioner);

        // calc jacobian
        this->equation->calcJacobian(this->phi[component], gamma, this->source->drivePattern->cols,
            this->source->measurementPattern->cols, component == 0 ? false : true,
            stream, this->jacobian);

        // calculate electrode voltage or current, depends on the type of source
        if (this->source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
            this->equation->update(gamma, alpha, gamma, stream);

            this->excitation->multiply(this->equation->systemMatrix,
                this->phi[component], handle, stream);
            this->excitation->scalarMultiply(this->equation->boundaryDescriptor->height,
                stream);

            this->applyMeasurementPattern(this->excitation, this->result,
                component == 0 ? false : true, handle, stream);
        }
        else {
            this->applyMeasurementPattern(this->phi[component], this->result,
                component == 0 ? false : true, handle, stream);
        }
    }

    // current source specific correction for jacobian matrix
    if (this->source->type == FEM::SourceDescriptor<dataType>::Type::Open) {
        this->jacobian->scalarMultiply(dataType(-1), stream);
    }

    // return mean step count
    if (steps != nullptr) {
        *steps = totalSteps;
    }

    return this->result;
}

// helper methods
template <
    template <class> class numericalSolverType,
    class equationType
>
void mpFlow::EIT::ForwardSolver<numericalSolverType, equationType>::applyMeasurementPattern(
    std::shared_ptr<numeric::Matrix<dataType> const> const source,
    std::shared_ptr<numeric::Matrix<dataType>> const result, bool const additiv,
    cublasHandle_t const handle, cudaStream_t const stream) const {
    // check input
    if (source == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::applyMeasurementPattern: source == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::applyMeasurementPattern: result == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::applyMeasurementPattern: handle == nullptr");
    }

    // set stream
    cublasSetStream(handle, stream);

    // add result
    dataType alpha = dataType(1), beta = additiv ? dataType(1) : dataType(0);
    numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->electrodesAttachmentMatrix->dataRows,
        this->source->drivePattern->cols, this->electrodesAttachmentMatrix->dataCols, &alpha,
        this->electrodesAttachmentMatrix->deviceData, this->electrodesAttachmentMatrix->dataRows,
        source->deviceData, source->dataRows, &beta,
        result->deviceData, result->dataRows);
}

template <
    template <class> class numericalSolverType,
    class equationType
>
void mpFlow::EIT::ForwardSolver<numericalSolverType, equationType>::applyMixedBoundaryCondition(
    std::shared_ptr<numeric::Matrix<dataType>> const excitationMatrix,
    std::shared_ptr<numeric::SparseMatrix<dataType>> const systemMatrix, cudaStream_t const stream) {
    // check input
    if (excitationMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::applyMixedBoundaryCondition: excitationMatrix == nullptr");
    }
    if (systemMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::applyMixedBoundaryCondition: systemMatrix == nullptr");
    }

    dim3 blocks(excitationMatrix->dataRows / numeric::matrix::blockSize,
        excitationMatrix->dataCols == 1 ? 1 : excitationMatrix->dataCols / numeric::matrix::blockSize);
    dim3 threads(numeric::matrix::blockSize,
        excitationMatrix->dataCols == 1 ? 1 : numeric::matrix::blockSize);

    forwardKernel::applyMixedBoundaryCondition<dataType>(blocks, threads, stream,
        excitationMatrix->deviceData, excitationMatrix->dataRows,
        systemMatrix->deviceColumnIds, systemMatrix->deviceValues);
}

// specialisation
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::EIT::ForwardSolver<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
