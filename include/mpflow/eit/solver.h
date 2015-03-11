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

#ifndef MPFLOW_INCLDUE_EIT_SOLVER_H
#define MPFLOW_INCLDUE_EIT_SOLVER_H

namespace mpFlow {
namespace EIT {
    // class for solving differential EIT
    template <
        class basisFunctionType,
        template <class, template <class> class> class numericalSolverType
    >
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<FEM::Equation<dtype::real, basisFunctionType, true>> equation,
            std::shared_ptr<FEM::SourceDescriptor> source, dtype::index components,
            dtype::index parallelImages, dtype::real regularizationFactor,
            cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solveDifferential(
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<numeric::Matrix<dtype::real>> solveAbsolute(cublasHandle_t handle,
            cudaStream_t stream);

        // member
        std::shared_ptr<ForwardSolver<basisFunctionType, numericalSolverType, true>> forwardSolver;
        std::shared_ptr<solver::Inverse<dtype::real, numeric::ConjugateGradient>> inverseSolver;
        std::shared_ptr<numeric::Matrix<dtype::real>> gamma;
        std::shared_ptr<numeric::Matrix<dtype::real>> dGamma;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> measurement;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> calculation;
    };
}
}

#endif
