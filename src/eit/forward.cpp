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
    class basisFunctionType,
    template <class, template <class> class> class numericalSolverType
>
mpFlow::EIT::ForwardSolver<basisFunctionType, numericalSolverType>::ForwardSolver(
    std::shared_ptr<FEM::Equation<dtype::real, basisFunctionType>> equation,
    std::shared_ptr<FEM::SourceDescriptor> source, dtype::index components,
    cublasHandle_t handle, cudaStream_t stream)
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
    this->numericalSolver = std::make_shared<numericalSolverType<
        dtype::real, numeric::SparseMatrix>>(
        this->equation->mesh->nodes->rows,
        this->source->drivePattern->cols + this->source->measurementPattern->cols, stream);

    // create matrices
    this->result = std::make_shared<numeric::Matrix<dtype::real>>(
        this->source->measurementPattern->cols, this->source->drivePattern->cols, stream);
    for (dtype::index component = 0; component < components; ++component) {
        this->phi.push_back(std::make_shared<numeric::Matrix<dtype::real>>(this->equation->mesh->nodes->rows,
            this->source->pattern->cols, stream));
    }
    this->excitation = std::make_shared<numeric::Matrix<dtype::real>>(this->equation->mesh->nodes->rows,
        this->source->pattern->cols, stream);
    this->jacobian = std::make_shared<numeric::Matrix<dtype::real>>(
        math::roundTo(this->source->measurementPattern->cols, numeric::matrix::block_size) *
        math::roundTo(this->source->drivePattern->cols, numeric::matrix::block_size),
        this->equation->mesh->elements->rows, stream, 0.0, false);

    // TODO: To be moved to new BoundaryValues class
    this->electrodesAttachmentMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->source->measurementPattern->cols,
        this->equation->mesh->nodes->rows, stream, 0.0, false);

    // apply mixed boundary conditions, if applicably
    if (this->source->type == FEM::sourceDescriptor::MixedSourceType) {
        forwardSolver::applyMixedBoundaryCondition(this->equation->excitationMatrix,
            this->equation->systemMatrix, stream);
    }

    cublasSetStream(handle, stream);
    dtype::real alpha = 1.0, beta = 0.0;
    if (numeric::cublasWrapper<dtype::real>::gemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        this->source->measurementPattern->dataCols,
        this->equation->excitationMatrix->dataRows,
        this->source->measurementPattern->dataRows, &alpha,
        this->source->measurementPattern->deviceData,
        this->source->measurementPattern->dataRows,
        this->equation->excitationMatrix->deviceData,
        this->equation->excitationMatrix->dataRows,
        &beta, this->electrodesAttachmentMatrix->deviceData,
        this->electrodesAttachmentMatrix->dataRows) != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("mpFlow::EIT::ForwardSolver: calc result calculation");
    }
}

// forward solving
template <
    class basisFunctionType,
    template <class, template <class> class> class numericalSolverType
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::ForwardSolver<basisFunctionType, numericalSolverType>::solve(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream, dtype::real tolerance) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: gamma == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::ForwardSolver::solve: handle == nullptr");
    }

    // calculate common excitation for all 2.5D model components
    for (dtype::index component = 0; component < this->phi.size(); ++component) {
        // 2.5D model constants
        dtype::real alpha = math::square(2.0 * component * M_PI / this->equation->mesh->height);
        dtype::real beta = component == 0 ? (1.0 / this->equation->mesh->height) :
            (2.0 * sin(component * M_PI * std::get<1>(this->equation->boundaryDescriptor->shapes[0]) / this->equation->mesh->height) /
                (component * M_PI * std::get<1>(this->equation->boundaryDescriptor->shapes[0])));

        // update system matrix for different 2.5D components
        this->equation->update(gamma, alpha, gamma, stream);

        if (this->source->type == FEM::sourceDescriptor::MixedSourceType) {
            forwardSolver::applyMixedBoundaryCondition(this->equation->excitationMatrix,
                this->equation->systemMatrix, stream);
        }

        this->excitation->multiply(this->equation->excitationMatrix,
            this->source->pattern, handle, stream);
        this->excitation->scalarMultiply(beta, stream);

        // solve linear system
        this->numericalSolver->solve(this->equation->systemMatrix,
            this->excitation, steps, nullptr, stream, this->phi[component],
            tolerance, component == 0 ? true : false);

        // calc jacobian
        this->equation->calcJacobian(this->phi[component], gamma, this->source->drivePattern->cols,
            this->source->measurementPattern->cols, component == 0 ? false : true,
            stream, this->jacobian);

        // calculate electrode voltage or current, depends on the type of source
        if (this->source->type == FEM::sourceDescriptor::MixedSourceType) {
            this->equation->update(gamma, alpha, gamma, stream);

            this->phi[component]->scalarMultiply(1.0 / (dtype::real)this->phi.size(), stream);
            this->excitation->multiply(this->equation->systemMatrix,
                this->phi[component], handle, stream);
            this->excitation->scalarMultiply(std::get<1>(this->equation->boundaryDescriptor->shapes[0]) /
                (dtype::real)this->phi.size(), stream);

            this->applyMeasurementPattern(this->excitation, this->result,
                component == 0 ? false : true, handle, stream);
        }
        else {
            this->applyMeasurementPattern(this->phi[component], this->result,
                component == 0 ? false : true, handle, stream);
        }
    }

    // current source specific correction for jacobian matrix
    if (this->source->type == FEM::sourceDescriptor::OpenSourceType) {
        this->jacobian->scalarMultiply(-1.0, stream);
    }

    return this->result;
}

// helper methods
template <
    class basisFunctionType,
    template <class, template <class> class> class numericalSolverType
>
void mpFlow::EIT::ForwardSolver<basisFunctionType, numericalSolverType>::applyMeasurementPattern(
    const std::shared_ptr<numeric::Matrix<dtype::real>> source,
    std::shared_ptr<numeric::Matrix<dtype::real>> result, bool additiv,
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (source == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMeasurementPattern: source == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMeasurementPattern: result == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMeasurementPattern: handle == nullptr");
    }

    // set stream
    cublasSetStream(handle, stream);

    // add result
    dtype::real alpha = 1.0f, beta = additiv ? 1.0 : 0.0;
    numeric::cublasWrapper<dtype::real>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->electrodesAttachmentMatrix->dataRows,
        this->source->drivePattern->cols, this->electrodesAttachmentMatrix->dataCols, &alpha,
        this->electrodesAttachmentMatrix->deviceData, this->electrodesAttachmentMatrix->dataRows,
        source->deviceData, source->dataRows, &beta,
        result->deviceData, result->dataRows);
}

void mpFlow::EIT::forwardSolver::applyMixedBoundaryCondition(
    std::shared_ptr<numeric::Matrix<dtype::real>> excitationMatrix,
    std::shared_ptr<numeric::SparseMatrix<dtype::real>> systemMatrix, cudaStream_t stream) {
    // check input
    if (excitationMatrix == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMixedBoundaryCondition: excitationMatrix == nullptr");
    }
    if (systemMatrix == nullptr) {
        throw std::invalid_argument("fastEIT::ForwardSolver::applyMixedBoundaryCondition: systemMatrix == nullptr");
    }

    dim3 blocks(excitationMatrix->dataRows / numeric::matrix::block_size,
        excitationMatrix->dataCols == 1 ? 1 : excitationMatrix->dataCols / numeric::matrix::block_size);
    dim3 threads(numeric::matrix::block_size,
        excitationMatrix->dataCols == 1 ? 1 : numeric::matrix::block_size);

    forwardKernel::applyMixedBoundaryCondition(blocks, threads, stream,
        excitationMatrix->deviceData, excitationMatrix->dataRows,
        systemMatrix->columnIds, systemMatrix->values);
}

// specialisation
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::basis::Linear, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::basis::Quadratic, mpFlow::numeric::ConjugateGradient>;
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::basis::Linear, mpFlow::numeric::BiCGSTAB>;
template class mpFlow::EIT::ForwardSolver<mpFlow::FEM::basis::Quadratic, mpFlow::numeric::BiCGSTAB>;
