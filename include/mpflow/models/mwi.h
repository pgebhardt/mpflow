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
            dataType const referenceWaveNumber, cublasHandle_t const handle,
            cudaStream_t const stream);

        // forward solving
        std::shared_ptr<numeric::Matrix<dataType> const> solve(
            std::shared_ptr<numeric::Matrix<dataType> const> const materialDistribution,
            cublasHandle_t const handle, cudaStream_t const stream,
            unsigned* const steps=nullptr);

        // member
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
        std::shared_ptr<FEM::Sources<dataType> const> const sources;
        std::shared_ptr<numeric::Matrix<dataType>> fields;
        dataType const referenceWaveNumber;
        
    private:
        std::shared_ptr<equationType> equation;
        std::shared_ptr<numericalSolverType<dataType>> numericalSolver;
        Eigen::Matrix<typename typeTraits::convertComplexType<dataType>::type,
            Eigen::Dynamic, Eigen::Dynamic> excitation;
        std::shared_ptr<numeric::Matrix<dataType>> alpha;
    };
}
}

#endif
