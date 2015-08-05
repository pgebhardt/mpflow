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
#include "mpflow/models/eit_kernel.h"

template <
    template <class> class numericalSolverType,
    class equationType
>
mpFlow::models::MWI<numericalSolverType, equationType>::MWI(
    std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<FEM::Sources<dataType> const> const sources,
    dataType const referenceWaveNumber, cublasHandle_t const handle,
    cudaStream_t const stream)
    : mesh(mesh), sources(sources), referenceWaveNumber(referenceWaveNumber) {
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
    this->fields = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->edges.rows(), this->sources->pattern->cols, stream);
    this->alpha = std::make_shared<numeric::Matrix<dataType>>(
        this->mesh->elements.rows(), 1, stream, 1.0);
        
    // create mass matrix
    auto const excitationMatrix = std::make_shared<numeric::Matrix<dataType>>(
        equation->excitationMatrix->rows, equation->excitationMatrix->cols, stream);
    excitationMatrix->copy(equation->excitationMatrix, stream);
    excitationMatrix->copyToHost(stream);
    cudaStreamSynchronize(stream);
    
    this->excitation = excitationMatrix->toEigen().matrix();
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
    cublasHandle_t const handle, cudaStream_t const stream, unsigned* const) {
    // check input
    if (materialDistribution == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::solve: materialDistribution == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::models::MWI::solve: handle == nullptr");
    }

    // update equation for new material distribution
    this->equation->update(this->alpha, dataType(1), materialDistribution, stream);
    
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
            dataType(0.0, 1.0) * length * this->referenceWaveNumber);
                
        // fix tangential field to zero on all boundary edges excepts the ports
        if (abs(this->sources->ports->edges - edgeIndex).minCoeff() != 0) {
            clearRow(equation->systemMatrix, edgeIndex);  
            equation->systemMatrix->setValue(edgeIndex, edgeIndex, dataType(1));  
        }
    }
    equation->systemMatrix->copyToDevice(stream);
    
    // convert system Matrix to eigen matrix
    auto const AMatrix = equation->systemMatrix->toMatrix(stream);
    AMatrix->copyToHost(stream);
    cudaStreamSynchronize(stream);       
    auto const A = AMatrix->toEigen().matrix().eval();
    
    // solve system using eigen
    auto const b = (this->excitation * this->sources->pattern->toEigen().matrix() *
        dataType(0.0, -1.0) * 2.0 * M_PI * constants::mu0).eval();
    auto const x = A.partialPivLu().solve(b).eval();

    // copy result back to Device
    this->fields = numeric::Matrix<dataType>::fromEigen(x.array(), stream);
    this->fields->copyToDevice(stream);
    
    return nullptr;
}

// specialisation
template class mpFlow::models::MWI<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<float>, mpFlow::FEM::basis::Edge, false>>;
template class mpFlow::models::MWI<mpFlow::numeric::BiCGSTAB,
    mpFlow::FEM::Equation<thrust::complex<double>, mpFlow::FEM::basis::Edge, false>>;