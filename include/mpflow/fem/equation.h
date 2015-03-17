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

#ifndef MPFLOW_INCLDUE_FEM_EQUATION_H
#define MPFLOW_INCLDUE_FEM_EQUATION_H

namespace mpFlow {
namespace FEM {
    // model class describing an elliptical differential equation
    template <
        class dataType_,
        class basisFunctionType_,
        bool logarithmic=true
    >
    class Equation {
    public:
        typedef basisFunctionType_ basisFunctionType;
        typedef dataType_ dataType;

        // constructor
        Equation(std::shared_ptr<numeric::IrregularMesh const> const mesh,
            std::shared_ptr<FEM::BoundaryDescriptor const> const boundaryDescriptor,
            dataType const referenceValue, cudaStream_t const stream);

        // init methods
        void initElementalMatrices(cudaStream_t const stream);
        void initExcitationMatrix(cudaStream_t const stream);
        void initJacobianCalculationMatrix(cudaStream_t const stream);

        void calcJacobian(std::shared_ptr<numeric::Matrix<dataType> const> const field,
            std::shared_ptr<numeric::Matrix<dataType> const> const factor,
            unsigned const driveCount, unsigned const measurmentCount, bool const additiv,
            cudaStream_t const stream, std::shared_ptr<numeric::Matrix<dataType>> result) const;

        void update(std::shared_ptr<numeric::Matrix<dataType> const> const alpha,
            dataType const k, std::shared_ptr<numeric::Matrix<dataType> const> const beta,
            cudaStream_t const stream);

        // member
        std::shared_ptr<numeric::IrregularMesh const> const mesh;
        std::shared_ptr<FEM::BoundaryDescriptor const> const boundaryDescriptor;
        std::shared_ptr<numeric::SparseMatrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> excitationMatrix;
        dataType const referenceValue;

    private:
        std::shared_ptr<numeric::Matrix<unsigned>> connectivityMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> sMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> rMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalSMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalRMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalJacobianMatrix;
        std::shared_ptr<numeric::Matrix<unsigned>> meshElements;
    };

    namespace equation {
        // update matrix
        template <
            class dataType,
            bool logarithmic
        >
        void updateMatrix(std::shared_ptr<numeric::Matrix<dataType> const> const elements,
            std::shared_ptr<numeric::Matrix<dataType> const> const gamma,
            std::shared_ptr<numeric::Matrix<unsigned> const> const connectivityMatrix,
            dataType const referenceValue, cudaStream_t const stream,
            std::shared_ptr<numeric::SparseMatrix<dataType>> matrix);
    }
}
}

#endif
