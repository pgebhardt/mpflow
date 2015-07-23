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

#ifndef MPFLOW_INCLDUE_MWI_EQUATION_H
#define MPFLOW_INCLDUE_MWI_EQUATION_H

namespace mpFlow {
namespace MWI {
    // model class describing an 2D TE Helmholz equation
    template <
        class dataType
    >
    class Equation {
    public:
        // constructor
        Equation(std::shared_ptr<numeric::IrregularMesh const> const mesh,
            cudaStream_t const stream);

        // init methods
        void initElementalMatrices(cudaStream_t const stream);
        void initJacobianCalculationMatrix(cudaStream_t const stream);

        void calcJacobian(
            std::shared_ptr<numeric::Matrix<dataType> const> const field,
            cudaStream_t const stream, std::shared_ptr<numeric::Matrix<dataType>> const jacobian) const;

        void update(std::shared_ptr<numeric::Matrix<dataType> const> const beta,
            dataType const k, cudaStream_t const stream);

        // member
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
        std::shared_ptr<numeric::SparseMatrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::Matrix<unsigned>> connectivityMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> sMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> rMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalSMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalRMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalJacobianMatrix;
        std::shared_ptr<numeric::Matrix<int>> edges;
    };
}
}

#endif
