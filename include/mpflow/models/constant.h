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

#ifndef MPFLOW_INCLDUE_MODELS_CONSTANT_H
#define MPFLOW_INCLDUE_MODELS_CONSTANT_H

namespace mpFlow {
namespace models {
    // model with constant jacobian
    template <
        class dataType_ = float
    >
    class Constant {
    public:
        typedef dataType_ dataType;

        // initialization
        Constant(std::shared_ptr<numeric::IrregularMesh const> const mesh,
            std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source,
            std::shared_ptr<numeric::Matrix<dataType>> const jacobian,
            dataType const referenceValue, cudaStream_t const stream);
        
        static std::shared_ptr<Constant<dataType>>
            fromConfig(json_value const& config, cublasHandle_t const handle,
            cudaStream_t const stream, std::string const path="./",
            std::shared_ptr<numeric::IrregularMesh const> const externalMesh=nullptr);

        // forward solving
        std::shared_ptr<numeric::Matrix<dataType> const> solve(
            std::shared_ptr<numeric::Matrix<dataType> const> const,
            cublasHandle_t const, cudaStream_t const, unsigned* const steps=nullptr);

        // member
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
        std::shared_ptr<FEM::SourceDescriptor<dataType> const> const source;
        std::shared_ptr<numeric::Matrix<dataType>> const jacobian;
        std::shared_ptr<numeric::Matrix<dataType>> result;
        dataType const referenceValue;

    private:
        std::shared_ptr<numeric::Matrix<dataType>> excitation;
    };
}
}

#endif
