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

#ifndef MPFLOW_INCLDUE_EIT_INVERSE_SOLVER_H
#define MPFLOW_INCLDUE_EIT_INVERSE_SOLVER_H

namespace mpFlow {
namespace EIT {
    // inverse solver class definition
    template <
        template <template <class> class> class numericalSolverType
    >
    class InverseSolver {
    public:
        // constructor
        InverseSolver(dtype::size elementCount, dtype::size measurementCount, dtype::index parallelImages,
            dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream);

    public:
        // inverse solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solve(
            const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement,
            dtype::size steps, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<numeric::Matrix<dtype::real>> gamma);

        // calc system matrix
        void calcSystemMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
            cublasHandle_t handle, cudaStream_t stream);

        // calc excitation
        void calcExcitation(const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement,
            cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<numericalSolverType<mpFlow::numeric::Matrix>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dtype::real>> difference;
        std::shared_ptr<numeric::Matrix<dtype::real>> zeros;
        std::shared_ptr<numeric::Matrix<dtype::real>> excitation;
        std::shared_ptr<numeric::Matrix<dtype::real>> systemMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> jacobianSquare;
        dtype::real regularizationFactor;
    };
}
}

#endif
