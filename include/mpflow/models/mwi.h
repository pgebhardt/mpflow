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

#ifndef MPFLOW_INCLDUE_MODELS_MWI_H
#define MPFLOW_INCLDUE_MODELS_MWI_H

namespace mpFlow {
namespace models {
    // MWI forward model
    template <
        template <class> class numericalSolverType = numeric::ConjugateGradient,
        class equationType_ = FEM::Equation<float, FEM::basis::Edge, false>
    >
    class MWI {
    public:
        typedef equationType_ equationType;
        typedef typename equationType::dataType dataType;
        static bool const logarithmic = equationType::logarithmic;
        
        // initialization
        MWI(std::shared_ptr<numeric::IrregularMesh const> const mesh,
            std::shared_ptr<FEM::Sources<dataType> const> const sources,
            double const frequency, double const height, double const portHeight,
            dataType const portMaterial, std::shared_ptr<numeric::Matrix<unsigned> const> const portElements,
            cublasHandle_t const handle, cudaStream_t const stream);

        // factories
#ifdef _JSON_H
        static std::shared_ptr<MWI<numericalSolverType, equationType>>
            fromConfig(json_value const& config, cublasHandle_t const handle,
            cudaStream_t const stream, std::string const path="./",
            std::shared_ptr<numeric::IrregularMesh const> const externalMesh=nullptr);
#endif

        // forward solving
        std::shared_ptr<numeric::Matrix<dataType> const> solve(
            std::shared_ptr<numeric::Matrix<dataType> const> const materialDistribution,
            cublasHandle_t const handle, cudaStream_t const stream,
            unsigned* const steps=nullptr);

        // member
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
        std::shared_ptr<FEM::Sources<dataType> const> const sources;
        std::shared_ptr<numeric::Matrix<dataType>> field;
        std::shared_ptr<numeric::Matrix<dataType>> result;
        std::shared_ptr<numeric::Matrix<dataType>> jacobian;
        double const frequency;
        double const height;
        double const portHeight;
        dataType const portMaterial;
        std::shared_ptr<numeric::Matrix<unsigned> const> portElements;

    private:
        std::shared_ptr<equationType> equation;
        std::shared_ptr<numericalSolverType<dataType>> numericalSolver;
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> alpha;
        std::shared_ptr<numeric::Matrix<dataType>> beta;
        std::shared_ptr<numeric::Matrix<dataType>> portsAttachmentMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> preconditioner;
    };
}
}

#endif
