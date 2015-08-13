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

#ifndef MPFLOW_INCLDUE_SOLVER_SOLVER_H
#define MPFLOW_INCLDUE_SOLVER_SOLVER_H

namespace mpFlow {
namespace solver {
    template <
        class forwardModelType = models::EIT<>,
        template <class> class numericalInverseSolverType = numeric::ConjugateGradient
    >
    class Solver {
    public:
        typedef typename forwardModelType::dataType dataType;

        // constructor
        Solver(std::shared_ptr<forwardModelType> const forwardModel,
            unsigned const parallelImages, cublasHandle_t const handle,
            cudaStream_t const stream);

        // factories
#ifdef _JSON_H
        static std::shared_ptr<Solver<forwardModelType, numericalInverseSolverType>>
            fromConfig(json_value const& config, cublasHandle_t const handle,
            cudaStream_t const stream, std::string const path="./",
            std::shared_ptr<numeric::IrregularMesh const> const externalMesh=nullptr);
#endif

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t const handle, cudaStream_t const stream);

        // solving
        std::shared_ptr<numeric::Matrix<dataType> const> solveDifferential(
            cublasHandle_t const handle, cudaStream_t const stream,
            unsigned const maxIterations=0, unsigned* const iterations=nullptr);
        std::shared_ptr<numeric::Matrix<dataType> const> solveAbsolute(
            unsigned const iterations, cublasHandle_t const handle,
            cudaStream_t const stream);

        // member
        std::shared_ptr<forwardModelType> const forwardModel;
        std::shared_ptr<solver::Inverse<dataType, numericalInverseSolverType>> inverseSolver;
        std::vector<std::shared_ptr<numeric::Matrix<dataType>>> measurement;
        std::vector<std::shared_ptr<numeric::Matrix<dataType>>> calculation;
        std::shared_ptr<numeric::Matrix<dataType>> referenceDistribution;
        std::shared_ptr<numeric::Matrix<dataType>> materialDistribution;
    };
}
}

#endif
