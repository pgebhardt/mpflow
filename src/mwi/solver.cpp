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

// create MWI
mpFlow::MWI::Solver::Solver(std::shared_ptr<numeric::IrregularMesh const> const mesh,
    std::shared_ptr<numeric::Matrix<thrust::complex<float>> const> const jacobian,
    unsigned const parallelImages, double const regularizationFactor,
    cublasHandle_t const handle, cudaStream_t const stream)
    : mesh(mesh), jacobian(jacobian) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Solver::Solver: mesh == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Solver::Solver: jacobian == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::MWI::Solver::Solver: handle == nullptr");
    }

    // create inverse solver
    this->inverseSolver = std::make_shared<solver::Inverse<thrust::complex<float>,
        numeric::BiCGSTAB>>(mesh, jacobian, parallelImages, handle, stream);
    this->inverseSolver->setRegularizationFactor(regularizationFactor, stream);

    // create matrices
    this->dGamma = std::make_shared<numeric::Matrix<thrust::complex<float>>>(
        mesh->elements.rows(), parallelImages, stream);
    for (unsigned image = 0; image < parallelImages; ++image) {
        this->measurement.push_back(std::make_shared<numeric::Matrix<thrust::complex<float>>>(
            jacobian->rows, 1, stream, 0.0, false));
        this->calculation.push_back(std::make_shared<numeric::Matrix<thrust::complex<float>>>(
            jacobian->rows, 1, stream, 0.0, false));
    }
}

// solve differential
std::shared_ptr<mpFlow::numeric::Matrix<thrust::complex<float>> const>
    mpFlow::MWI::Solver::solveDifferential(
    cublasHandle_t const handle, cudaStream_t const stream,
    unsigned const maxIterations) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::MWI::Solver::solveDifferential: handle == nullptr");
    }

    // solve
    this->inverseSolver->solve(this->calculation, this->measurement,
        maxIterations, handle, stream, this->dGamma);

    return this->dGamma;
}
