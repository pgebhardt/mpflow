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

#ifndef MPFLOW_INCLDUE_MODELS_EIT_H
#define MPFLOW_INCLDUE_MODELS_EIT_H

namespace mpFlow {
namespace models {
    // 2.5D EIT forward model
    template <
        template <class> class numericalSolverType = numeric::ConjugateGradient,
        class equationType_ = FEM::Equation<float, FEM::basis::Linear, false>
    >
    class EIT {
    public:
        typedef equationType_ equationType;
        typedef typename equationType::dataType dataType;
        static bool const logarithmic = equationType::logarithmic;

        // initialization
        EIT(std::shared_ptr<numeric::IrregularMesh const> const mesh,
            std::shared_ptr<FEM::Sources<dataType> const> const sources,
            dataType const referenceValue, cublasHandle_t const handle, cudaStream_t const stream,
            unsigned const components=1, double const height=1.0, double const portHeight=1.0);

        // factories
#ifdef _JSON_H
        static std::shared_ptr<EIT<numericalSolverType, equationType>>
            fromConfig(json_value const& config, cublasHandle_t const handle,
            cudaStream_t const stream, std::string const path="./",
            std::shared_ptr<numeric::IrregularMesh const> const externalMesh=nullptr);
#endif

        // forward solving
        std::shared_ptr<numeric::Matrix<dataType> const> solve(
            std::shared_ptr<numeric::Matrix<dataType> const> const materialDistribution,
            cublasHandle_t const handle, cudaStream_t const stream,
            unsigned* const steps=nullptr);

        // helper methods
        void applyMeasurementPattern(std::shared_ptr<numeric::Matrix<dataType> const> const sources,
            std::shared_ptr<numeric::Matrix<dataType>> const result, bool const additiv,
            cublasHandle_t const handle, cudaStream_t const stream) const;
        static void applyMixedBoundaryCondition(std::shared_ptr<numeric::Matrix<dataType>> const excitationMatrix,
            std::shared_ptr<numeric::SparseMatrix<dataType>> const systemMatrix, cudaStream_t const stream);

        // member
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
        std::shared_ptr<FEM::Sources<dataType> const> const sources;
        dataType const referenceValue;
        double const height;
        double const portHeight;
        std::vector<std::shared_ptr<numeric::Matrix<dataType>>> fields;
        std::shared_ptr<numeric::Matrix<dataType>> field;
        std::shared_ptr<numeric::Matrix<dataType>> result;
        std::shared_ptr<numeric::Matrix<dataType>> jacobian;

    private:
        std::shared_ptr<equationType> equation;
        std::shared_ptr<numericalSolverType<dataType>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> electrodesAttachmentMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> preconditioner;
    };
}
}

#endif
