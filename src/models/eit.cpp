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

#include "json.h"
#include "mpflow/mpflow.h"
#include "mpflow/models/eit_kernel.h"

template <
    template <class> class numericalSolverType,
    class equationType
>
mpFlow::models::EIT<numericalSolverType, equationType>::EIT(
    std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<FEM::Sources<dataType> const> const sources,
    dataType const referenceValue, cublasHandle_t const handle, cudaStream_t const stream,
    unsigned const components, double const height, double const portHeight) :
    mesh(mesh), sources(sources), referenceValue(referenceValue), height(height), portHeight(portHeight) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::EIT: mesh == nullptr");
    }
    if (sources == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::EIT: sources == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::EIT: handle == nullptr");
    }
    if (components < 1) {
        throw std::invalid_argument("mpFlow::models::EIT::EIT: components < 1");
    }
    if (height <= 0.0) {
        throw std::invalid_argument("mpFlow::models::EIT::EIT: height <= 0.0");
    }
    if (portHeight <= 0.0) {
        throw std::invalid_argument("mpFlow::models::EIT::EIT: portHeight <= 0.0");
    }

    // create FEM equation
    this->equation = std::make_shared<equationType>(this->mesh,
        this->sources->ports, this->referenceValue, true, stream);

    // create numericalSolver solver
    this->numericalSolver = std::make_shared<numericalSolverType<dataType>>(
        this->mesh->nodes.rows(),
        this->sources->drivePattern->cols + this->sources->measurementPattern->cols, stream);

    // create matrices
    this->result = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->cols, this->sources->drivePattern->cols, stream);
    for (unsigned component = 0; component < components; ++component) {
        this->fields.push_back(std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes.rows(),
            this->sources->pattern->cols, stream,
            sources->type == FEM::Sources<dataType>::Type::Fixed ? dataType(1) : dataType(0)));

        if (component == 0) {
            this->field = this->fields[0];
        }
    }
    this->excitation = std::make_shared<numeric::Matrix<dataType>>(this->mesh->nodes.rows(),
        this->sources->pattern->cols, stream);
    this->jacobian = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->dataCols * this->sources->drivePattern->dataCols,
        this->mesh->elements.rows(), stream, 0.0, false);
    this->preconditioner = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->equation->systemMatrix->rows, this->equation->systemMatrix->cols, stream);

    // create matrix to calculate system excitation from electrode excitation
    this->electrodesAttachmentMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->cols,
        this->mesh->nodes.rows(), stream, 0.0, false);

    if (this->sources->type == FEM::Sources<dataType>::Type::Fixed) {
        applyMixedBoundaryCondition(this->equation->excitationMatrix, this->equation->systemMatrix, stream);
    }

    this->electrodesAttachmentMatrix->multiply(this->sources->measurementPattern,
        this->equation->excitationMatrix, handle, stream, CUBLAS_OP_T, CUBLAS_OP_T);
}

template <
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::models::EIT<numericalSolverType, equationType>>
    mpFlow::models::EIT<numericalSolverType, equationType>::fromConfig(
    json_value const& config, cublasHandle_t const handle, cudaStream_t const stream,
    std::string const path, std::shared_ptr<numeric::IrregularMesh const> const externalMesh) {
    // load mesh from config
    auto const mesh = externalMesh != nullptr ? externalMesh :
        numeric::IrregularMesh::fromConfig(config["mesh"], config["ports"], stream, path);

    // load ports descriptor from config
    auto const meshPath = config["mesh"].type == json_string ? std::string(config["mesh"]) :
        (config["mesh"]["path"].type == json_string ? std::string(config["mesh"]["path"]) : std::string("mesh"));
    auto const ports = FEM::Ports::fromConfig(config["ports"], mesh, stream, path, meshPath);

    // load sources from config
    auto const sources = FEM::Sources<dataType>::fromConfig(
        config["source"], ports, stream);

    // read out reference value
    auto const referenceValue = config["material"].type == json_object ?
        jsonHelper::parseNumericValue<dataType>(config["material"]["referenceValue"], 1.0) :
        jsonHelper::parseNumericValue<dataType>(config["material"], 1.0);

    // read 2.5D stuff
    auto const componentsCount = std::max(1, (int)config["2.5D"]["components"].u.integer);
    auto const height = config["2.5D"]["height"].type != json_none ? config["2.5D"]["height"].u.dbl : 1.0;
    auto const portHeight = config["ports"]["height"].type != json_none ? config["ports"]["height"].u.dbl : 1.0;

    // create forward model
    return std::make_shared<EIT<numericalSolverType, equationType>>(mesh, sources,
        referenceValue, handle, stream, componentsCount, height, portHeight);
}

// forward solving
template <
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::models::EIT<numericalSolverType, equationType>::solve(
    std::shared_ptr<numeric::Matrix<dataType> const> const materialDistribution,
    cublasHandle_t const handle, cudaStream_t const stream, unsigned* const steps) {
    // check input
    if (materialDistribution == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::solve: materialDistribution == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::solve: handle == nullptr");
    }

    // calculate common excitation for all 2.5D model components
    unsigned totalSteps = 0;
    for (unsigned component = 0; component < this->fields.size(); ++component) {
        // 2.5D model constants
        dataType const alpha = math::square(2.0 * component * M_PI / this->height);
        dataType const beta = component == 0 ? (1.0 / this->height) :
            (2.0 * sin(component * M_PI * this->portHeight / this->height) /
                (component * M_PI * this->portHeight));

        // update system matrix for different 2.5D components
        this->equation->update(materialDistribution, alpha, materialDistribution, stream);
        if (this->sources->type == FEM::Sources<dataType>::Type::Fixed) {
            applyMixedBoundaryCondition(this->equation->excitationMatrix, this->equation->systemMatrix, stream);
        }

        // create system excitation
        this->excitation->multiply(this->equation->excitationMatrix,
            this->sources->pattern, handle, stream);
        if (this->sources->type == FEM::Sources<dataType>::Type::Open) {
            this->excitation->scalarMultiply(beta, stream);
        }

        // solve linear system
        this->fields[component]->fill(sources->type == FEM::Sources<dataType>::Type::Fixed ? dataType(1) : dataType(0), stream);
        numeric::preconditioner::diagonal<dataType>(this->equation->systemMatrix, stream, this->preconditioner);
        totalSteps += this->numericalSolver->solve(this->equation->systemMatrix,
            this->excitation, nullptr, stream, this->fields[component], this->preconditioner, 0,
            (component == 0 && this->sources->type == FEM::Sources<dataType>::Type::Open) ? true : false);

        // calc jacobian
        this->equation->calcJacobian(this->fields[component], materialDistribution, this->sources->drivePattern->cols,
            this->sources->measurementPattern->cols, component == 0 ? false : true,
            stream, this->jacobian);

        // calculate electrode voltage or current, depends on the type of sources
        if (this->sources->type == FEM::Sources<dataType>::Type::Fixed) {
            this->equation->update(materialDistribution, alpha, materialDistribution, stream);

            this->excitation->multiply(this->equation->systemMatrix,
                this->fields[component], handle, stream);
            this->excitation->scalarMultiply(this->height, stream);

            this->applyMeasurementPattern(this->excitation, this->result,
                component == 0 ? false : true, handle, stream);
        }
        else {
            this->applyMeasurementPattern(this->fields[component], this->result,
                component == 0 ? false : true, handle, stream);
        }
    }

    // current sources specific correction for jacobian matrix
    this->jacobian->scalarMultiply(this->height, stream);
    if (this->sources->type == FEM::Sources<dataType>::Type::Open) {
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
void mpFlow::models::EIT<numericalSolverType, equationType>::applyMeasurementPattern(
    std::shared_ptr<numeric::Matrix<dataType> const> const sources,
    std::shared_ptr<numeric::Matrix<dataType>> const result, bool const additiv,
    cublasHandle_t const handle, cudaStream_t const stream) const {
    // check input
    if (sources == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::applyMeasurementPattern: sources == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::applyMeasurementPattern: result == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::applyMeasurementPattern: handle == nullptr");
    }

    // set stream
    cublasSetStream(handle, stream);

    // add result
    dataType alpha = dataType(1), beta = additiv ? dataType(1) : dataType(0);
    numeric::cublasWrapper<dataType>::gemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->electrodesAttachmentMatrix->dataRows,
        this->sources->drivePattern->cols, this->electrodesAttachmentMatrix->dataCols, &alpha,
        this->electrodesAttachmentMatrix->deviceData, this->electrodesAttachmentMatrix->dataRows,
        sources->deviceData, sources->dataRows, &beta,
        result->deviceData, result->dataRows);
}

template <
    template <class> class numericalSolverType,
    class equationType
>
void mpFlow::models::EIT<numericalSolverType, equationType>::applyMixedBoundaryCondition(
    std::shared_ptr<numeric::Matrix<dataType>> const excitationMatrix,
    std::shared_ptr<numeric::SparseMatrix<dataType>> const systemMatrix, cudaStream_t const stream) {
    // check input
    if (excitationMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::applyMixedBoundaryCondition: excitationMatrix == nullptr");
    }
    if (systemMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::models::EIT::applyMixedBoundaryCondition: systemMatrix == nullptr");
    }

    dim3 blocks(excitationMatrix->dataRows / numeric::matrix::blockSize,
        excitationMatrix->dataCols == 1 ? 1 : excitationMatrix->dataCols / numeric::matrix::blockSize);
    dim3 threads(numeric::matrix::blockSize,
        excitationMatrix->dataCols == 1 ? 1 : numeric::matrix::blockSize);

    eitKernel::applyMixedBoundaryCondition<dataType>(blocks, threads, stream,
        excitationMatrix->deviceData, excitationMatrix->dataRows,
        systemMatrix->deviceColumnIds, systemMatrix->deviceValues);
}

// specialisation
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<float, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<double, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::ConjugateGradient,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, true>>;
template class mpFlow::models::EIT<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Linear, false>>;
