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
        template <class, template <class> class> class numericalSolverType
    >
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<EIT::ForwardSolver<>::equationType> equation,
            std::shared_ptr<FEM::SourceDescriptor<float>> source, unsigned components,
            unsigned parallelImages, double regularizationFactor,
            cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<numeric::Matrix<float>> solveDifferential(
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<numeric::Matrix<float>> solveAbsolute(cublasHandle_t handle,
            cudaStream_t stream);

        // member
        std::shared_ptr<ForwardSolver<numericalSolverType>> forwardSolver;
        std::shared_ptr<solver::Inverse<float, numeric::ConjugateGradient>> inverseSolver;
        std::shared_ptr<numeric::Matrix<float>> gamma;
        std::shared_ptr<numeric::Matrix<float>> dGamma;
        std::vector<std::shared_ptr<numeric::Matrix<float>>> measurement;
        std::vector<std::shared_ptr<numeric::Matrix<float>>> calculation;
    };
}
}

#endif
