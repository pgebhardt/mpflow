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

#ifndef MPFLOW_INCLDUE_MWI_SOLVER_H
#define MPFLOW_INCLDUE_MWI_SOLVER_H

namespace mpFlow {
namespace MWI {
    // class for solving differential MWI
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<numeric::IrregularMesh> mesh,
            std::shared_ptr<numeric::Matrix<dtype::complex>> jacobian,
            dtype::index parallelImages, dtype::real regularizationFactor,
            cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<numeric::Matrix<dtype::complex>> solveDifferential(
            cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<solver::Inverse<dtype::complex, numeric::ConjugateGradient>> inverseSolver;
        std::shared_ptr<numeric::IrregularMesh> mesh;
        std::shared_ptr<numeric::Matrix<dtype::complex>> jacobian;
        std::shared_ptr<numeric::Matrix<dtype::complex>> dGamma;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::complex>>> measurement;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::complex>>> calculation;
    };
}
}

#endif
