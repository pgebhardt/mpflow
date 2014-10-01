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

#ifndef MPFLOW_INCLDUE_EIT_FORWARDSOLVER_H
#define MPFLOW_INCLDUE_EIT_FORWARDSOLVER_H

namespace mpFlow {
namespace EIT {
    // forward solver class definition
    template <
        class basisFunctionType,
        template <template <class> class> class numericalSolverType
    >
    class ForwardSolver {
    public:
        // initialization
        ForwardSolver(std::shared_ptr<Equation<basisFunctionType>> equation,
            std::shared_ptr<FEM::SourceDescriptor> source, dtype::index components,
            cublasHandle_t handle, cudaStream_t stream);

        // apply pattern
        void applyMeasurementPattern(const std::shared_ptr<numeric::Matrix<dtype::real>> source,
            std::shared_ptr<numeric::Matrix<dtype::real>> result, bool additiv,
            cublasHandle_t handle, cudaStream_t stream);

        // forward solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solve(
            const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, dtype::size steps,
            cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<numericalSolverType<mpFlow::numeric::SparseMatrix>> numericalSolver;
        std::shared_ptr<Equation<basisFunctionType>> equation;
        std::shared_ptr<FEM::SourceDescriptor> source;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> phi;
        std::shared_ptr<numeric::Matrix<dtype::real>> excitation;
        std::shared_ptr<numeric::Matrix<dtype::real>> result;
        std::shared_ptr<numeric::Matrix<dtype::real>> jacobian;
        std::shared_ptr<numeric::Matrix<dtype::real>> electrodesAttachmentMatrix;
    };
}
}

#endif
