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

#ifndef MPFLOW_INCLDUE_SOLVER_INVERSE_H
#define MPFLOW_INCLDUE_SOLVER_INVERSE_H

namespace mpFlow {
namespace solver {
    // inverse solver class definition
    template <
        class dataType,
        template <class, template <class> class> class numericalSolverType
    >
    class Inverse {
    public:
        enum RegularizationType {
            diagonal,
            square
        };

        // constructor
        Inverse(unsigned elementCount, unsigned measurementCount, unsigned parallelImages,
            dataType regularizationFactor, cublasHandle_t handle, cudaStream_t stream);

    public:
        // inverse solving
        std::shared_ptr<numeric::Matrix<dataType>> solve(
            const std::shared_ptr<numeric::Matrix<dataType>> jacobian,
            const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& calculation,
            const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& measurement,
            unsigned steps, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<numeric::Matrix<dataType>> gamma);

        // calc system matrix
        void calcSystemMatrix(const std::shared_ptr<numeric::Matrix<dataType>> jacobian,
            RegularizationType regularizationType, cublasHandle_t handle, cudaStream_t stream);

        // calc excitation
        void calcExcitation(const std::shared_ptr<numeric::Matrix<dataType>> jacobian,
            const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& calculation,
            const std::vector<std::shared_ptr<numeric::Matrix<dataType>>>& measurement,
            cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<numericalSolverType<dataType, mpFlow::numeric::Matrix>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dataType>> difference;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> jacobianSquare;
        dataType regularizationFactor;
    };
}
}

#endif
