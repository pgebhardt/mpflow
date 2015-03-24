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
        template <class> class numericalSolverType = numeric::ConjugateGradient,
        class equationType_ = FEM::Equation<float, FEM::basis::Linear, true>
    >
    class ForwardSolver {
    public:
        typedef equationType_ equationType;
        typedef typename equationType::dataType dataType;

        // initialization
        ForwardSolver(std::shared_ptr<equationType> const equation,
            std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
            unsigned const components, cublasHandle_t const handle, cudaStream_t const stream);

        // forward solving
        std::shared_ptr<numeric::Matrix<dataType> const> solve(
            std::shared_ptr<numeric::Matrix<dataType> const> const gamma,
            cublasHandle_t const handle, cudaStream_t const stream,
            double const tolerance=1e-6, unsigned* const steps=nullptr);

        // helper methods
        void applyMeasurementPattern(std::shared_ptr<numeric::Matrix<dataType> const> const source,
            std::shared_ptr<numeric::Matrix<dataType>> const result, bool const additiv,
            cublasHandle_t const handle, cudaStream_t const stream) const;
        static void applyMixedBoundaryCondition(std::shared_ptr<numeric::Matrix<dataType>> const excitationMatrix,
            std::shared_ptr<numeric::SparseMatrix<dataType>> const systemMatrix, cudaStream_t const stream);
        static void createPreconditioner(std::shared_ptr<numeric::SparseMatrix<dataType> const> const systemMatrix,
            std::shared_ptr<numeric::SparseMatrix<dataType>> const preconditioner, cudaStream_t const stream);

        // member
        std::shared_ptr<equationType> const equation;
        std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source;
        std::vector<std::shared_ptr<numeric::Matrix<dataType>>> phi;
        std::shared_ptr<numeric::Matrix<dataType>> result;
        std::shared_ptr<numeric::Matrix<dataType>> jacobian;

    private:
        std::shared_ptr<numericalSolverType<dataType>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> electrodesAttachmentMatrix;
    };
}
}

#endif
