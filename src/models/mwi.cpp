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
mpFlow::models::MWI<numericalSolverType, equationType>::MWI(
    std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<FEM::Sources<dataType> const> const sources,
    double const frequency, double const height, dataType const referenceValue, double const portHeight,
    dataType const portMaterial, std::shared_ptr<numeric::Matrix<unsigned> const> const portElements,
    cublasHandle_t const handle, cudaStream_t const stream)
    : mesh(mesh), sources(sources), frequency(frequency), height(height), referenceValue(referenceValue),
    portHeight(portHeight), portMaterial(portMaterial), portElements(portElements) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::MWI: mesh == nullptr");
    }
    if (sources == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::MWI: sources == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::MWI: handle == nullptr");
    }

    // create FEM equation
    this->equation = std::make_shared<equationType>(this->mesh,
        this->sources->ports, dataType(1), false, stream);

    // create numericalSolver solver
    this->numericalSolver = std::make_shared<numericalSolverType<dataType>>(
        this->mesh->edges.rows(),
        this->sources->drivePattern->cols + this->sources->measurementPattern->cols, stream);

    // create matrices
    this->result = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->cols, this->sources->drivePattern->cols, stream);
    this->field = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->edges.rows(), this->sources->pattern->cols, stream);
    this->excitation = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->edges.rows(), this->sources->pattern->cols, stream);
    this->jacobian = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->dataCols * this->sources->drivePattern->dataCols,
        this->mesh->elements.rows(), stream);
    this->preconditioner = std::make_shared<numeric::SparseMatrix<dataType>>(
        this->equation->systemMatrix->rows, this->equation->systemMatrix->cols, stream);
    this->alpha = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements.rows(), 1, stream, 1.0);
    this->beta = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements.rows(), 1, stream);

    // create matrix to calculate system excitation from ports excitation
    this->portsAttachmentMatrix = std::make_shared<numeric::Matrix<dataType>>(
        this->sources->measurementPattern->cols,
        this->mesh->edges.rows(), stream, 0.0, false);
    this->portsAttachmentMatrix->multiply(this->sources->measurementPattern,
        this->equation->excitationMatrix, handle, stream, CUBLAS_OP_T, CUBLAS_OP_T);
}

template <
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::models::MWI<numericalSolverType, equationType>>
    mpFlow::models::MWI<numericalSolverType, equationType>::fromConfig(
    json_value const& config, cublasHandle_t const handle, cudaStream_t const stream,
    std::string const path, std::shared_ptr<numeric::IrregularMesh const> const externalMesh) {
    // load mesh from config
    auto const mesh = externalMesh != nullptr ? externalMesh :
        numeric::IrregularMesh::fromConfig(config["mesh"], config["ports"], stream, path);

    // load ports descriptor from config
    auto const ports = FEM::Ports::fromConfig(config["ports"], mesh, stream, path);

    // load sources from config
    auto const sources = FEM::Sources<dataType>::fromConfig(
        config["source"], ports, stream);

    // read out mwi specific config
    auto const frequency = config["mwi"]["frequency"].u.dbl;
    auto const height = config["mwi"]["height"].type == json_double ? config["mwi"]["height"].u.dbl : 1.0;
    auto const portHeight = config["ports"]["height"].type == json_double ? config["ports"]["height"].u.dbl : 1.0;
    auto const portMaterial = jsonHelper::parseNumericValue<dataType>(config["ports"]["material"], 1.0);

    auto const portElements = [=](json_value const& config) {
        if (config.type == json_string) {
            return numeric::Matrix<unsigned>::loadtxt(
                str::format("%s/%s")(path, std::string(config)), stream);
        }
        else {
            return std::shared_ptr<numeric::Matrix<unsigned>>(nullptr);
        }
    }(config["ports"]["elements"]);

    // read out reference value
    auto const referenceValue = config["material"].type == json_object ?
        jsonHelper::parseNumericValue<dataType>(config["material"]["referenceValue"], 1.0) :
        jsonHelper::parseNumericValue<dataType>(config["material"], 1.0);

    // create forward model
    return std::make_shared<MWI<numericalSolverType, equationType>>(mesh, sources,
        frequency, height, referenceValue, portHeight, portMaterial, portElements, handle, stream);
}

template <class dataType>
void clearRow(std::shared_ptr<mpFlow::numeric::SparseMatrix<dataType>> const matrix,
    unsigned const row) {
    for (unsigned i = 0; i < mpFlow::numeric::sparseMatrix::blockSize; ++i) {
        matrix->hostColumnIds[row * mpFlow::numeric::sparseMatrix::blockSize + i] = mpFlow::constants::invalidIndex;
        matrix->hostValues[row * mpFlow::numeric::sparseMatrix::blockSize + i] = dataType(0);
    }
}

// forward solving
template <
    template <class> class numericalSolverType,
    class equationType
>
std::shared_ptr<mpFlow::numeric::Matrix<typename equationType::dataType> const>
    mpFlow::models::MWI<numericalSolverType, equationType>::solve(
    std::shared_ptr<numeric::Matrix<dataType> const> const materialDistribution,
    cublasHandle_t const handle, cudaStream_t const stream, unsigned* const steps) {
    // check input
    if (materialDistribution == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::solve: materialDistribution == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::solve: handle == nullptr");
    }

    // calculate material parameter
    dataType const k0 = 2.0 * M_PI * this->frequency / constants::c0;
    dataType const kBc = M_PI / this->height;
    dataType const kPc = M_PI / this->portHeight;
    dataType const kP = sqrt(math::square(k0) * this->portMaterial - math::square(kPc)); 

    this->beta->copy(materialDistribution, stream);
    this->beta->scalarMultiply(math::square(k0) * this->referenceValue, stream);
    this->beta->add(-math::square(kBc), stream);

    if (this->portElements) {
        this->beta->setIndexedElements(this->portElements,
            math::square(kP), stream);
    }

    // update equation for new material distribution
    this->equation->update(this->alpha, -dataType(1), this->beta, stream);

    // make system matrix available on host for post processing
    this->equation->systemMatrix->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // apply first order ABC
    for (unsigned i = 0; i < this->mesh->boundary.rows(); ++i) {
        // get boundary edge
        auto const edgeIndex = this->mesh->boundary(i);
        auto const edge = this->mesh->edges.row(edgeIndex).eval();

        // calculate length of edge
        dataType const length = std::sqrt(
            (this->mesh->nodes.row(edge(1)) - this->mesh->nodes.row(edge(0)))
            .square().sum());

        this->equation->systemMatrix->setValue(edgeIndex, edgeIndex,
            this->equation->systemMatrix->getValue(edgeIndex, edgeIndex) +
            dataType(0.0, 1.0) * length * kP);

        // fix tangential field to zero on all boundary edges excepts the ports
        if (abs(this->sources->ports->edges - edgeIndex).minCoeff() != 0) {
            clearRow(equation->systemMatrix, edgeIndex);
            equation->systemMatrix->setValue(edgeIndex, edgeIndex, dataType(1));
        }
    }
    equation->systemMatrix->copyToDevice(stream);

    // create system excitation
    this->excitation->multiply(this->equation->excitationMatrix,
        this->sources->pattern, handle, stream);
    this->excitation->scalarMultiply(dataType(0.0, -2.0 * M_PI * this->frequency * constants::mu0), stream);

    // solve linear system
    numeric::preconditioner::diagonal<dataType>(this->equation->systemMatrix, stream, this->preconditioner);
    unsigned const _steps = this->numericalSolver->solve(this->equation->systemMatrix, this->excitation, nullptr,
        stream, this->field, this->preconditioner);

    // calc jacobian
    this->equation->calcJacobian(this->field, materialDistribution, this->sources->drivePattern->cols,
        this->sources->measurementPattern->cols, false, stream, this->jacobian);
    this->jacobian->scalarMultiply(
        dataType(0.0, 2.0 * M_PI * this->frequency * constants::epsilon0) * this->referenceValue, stream);

/*    // don't use reflection data
    this->jacobian->copyToHost(stream);
    cudaStreamSynchronize(stream);

    unsigned const dim = std::sqrt(this->jacobian->rows);
    for (unsigned i = 0; i < dim; ++i)
    for (unsigned e = 0; e < this->jacobian->cols; ++e) {
        (*this->jacobian)(i * dim + i, e) = dataType(0);
    }
    this->jacobian->copyToDevice(stream);*/

    // calculate port parameter
    this->result->multiply(this->portsAttachmentMatrix, this->field, handle, stream);

    if (steps != nullptr) {
        *steps = _steps;
    }

    return this->result;
}

// specialisation
template class mpFlow::models::MWI<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Edge, false>>;
template class mpFlow::models::MWI<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Edge, false>>;
template class mpFlow::models::MWI<mpFlow::numeric::CPUSolver,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Edge, false>>;
template class mpFlow::models::MWI<mpFlow::numeric::CPUSolver,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Edge, false>>;
template class mpFlow::models::MWI<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Edge, true>>;
template class mpFlow::models::MWI<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Edge, true>>;
template class mpFlow::models::MWI<mpFlow::numeric::CPUSolver,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Edge, true>>;
template class mpFlow::models::MWI<mpFlow::numeric::CPUSolver,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Edge, true>>;
