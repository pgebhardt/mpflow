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
        template <class> class numericalSolverType
    >
    class Inverse {
    public:
        enum RegularizationType {
            diagonal,
            square
        };

        // constructor
        Inverse(unsigned const elementCount, unsigned const measurementCount,
            unsigned const parallelImages, dataType const regularizationFactor,
            cublasHandle_t const handle, cudaStream_t const stream);

    public:
        // inverse solving
        void solve(std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
            std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
            std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
            unsigned const steps, cublasHandle_t const handle, cudaStream_t const stream,
            std::shared_ptr<numeric::Matrix<dataType>> gamma);

        // calc system matrix
        void calcSystemMatrix(std::shared_ptr<numeric::Matrix<dataType> const> const jacobian,
            RegularizationType const regularizationType, cublasHandle_t const handle,
            cudaStream_t const stream);

        // calc excitation
        void calcExcitation(std::shared_ptr<numeric::Matrix<dataType> const> jacobian,
            std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& calculation,
            std::vector<std::shared_ptr<numeric::Matrix<dataType>>> const& measurement,
            cublasHandle_t const handle, cudaStream_t const stream);

        // member
        dataType regularizationFactor;

    private:
        std::shared_ptr<numericalSolverType<dataType>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dataType>> difference;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> jacobianSquare;
    };
}
}

#endif
