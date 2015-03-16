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
        template <class, template <class> class> class numericalSolverType = numeric::ConjugateGradient,
        class equationType_ = FEM::Equation<float, FEM::basis::Linear, true>
    >
    class ForwardSolver {
    public:
        typedef equationType_ equationType;
        typedef typename equationType::dataType dataType;

        // initialization
        ForwardSolver(std::shared_ptr<equationType> equation,
            std::shared_ptr<FEM::SourceDescriptor<dataType>> source,
            unsigned components, cublasHandle_t handle, cudaStream_t stream);

        // forward solving
        std::shared_ptr<numeric::Matrix<dataType>> solve(
            const std::shared_ptr<numeric::Matrix<dataType>> gamma,
            cublasHandle_t handle, cudaStream_t stream, double tolerance=1e-6,
            unsigned* steps=nullptr);

        // helper methods
        void applyMeasurementPattern(const std::shared_ptr<numeric::Matrix<dataType>> source,
            std::shared_ptr<numeric::Matrix<dataType>> result, bool additiv,
            cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<numericalSolverType<dataType,
            mpFlow::numeric::SparseMatrix>> numericalSolver;
        std::shared_ptr<equationType> equation;
        std::shared_ptr<FEM::SourceDescriptor<dataType>> source;
        std::vector<std::shared_ptr<numeric::Matrix<dataType>>> phi;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> result;
        std::shared_ptr<numeric::Matrix<dataType>> jacobian;
        std::shared_ptr<numeric::Matrix<dataType>> electrodesAttachmentMatrix;
    };

    namespace forwardSolver {
        template <
            class dataType
        >
        void applyMixedBoundaryCondition(std::shared_ptr<numeric::Matrix<dataType>> excitationMatrix,
            std::shared_ptr<numeric::SparseMatrix<dataType>> systemMatrix, cudaStream_t stream);
    }
}
}

#endif
