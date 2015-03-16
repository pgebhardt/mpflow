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
        Equation(std::shared_ptr<numeric::IrregularMesh> mesh,
            std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor,
            dataType referenceValue, cudaStream_t stream);

        // init methods
        void initElementalMatrices(cudaStream_t stream);
        void initExcitationMatrix(cudaStream_t stream);
        void initJacobianCalculationMatrix(cudaStream_t stream);

        void calcJacobian(const std::shared_ptr<numeric::Matrix<dataType>> field,
            const std::shared_ptr<numeric::Matrix<dataType>> factor,
            unsigned driveCount, unsigned measurmentCount, bool additiv,
            cudaStream_t stream, std::shared_ptr<numeric::Matrix<dataType>> result);

        void update(const std::shared_ptr<numeric::Matrix<dataType>> alpha,
            const dataType k, const std::shared_ptr<numeric::Matrix<dataType>> beta,
            cudaStream_t stream);

        // member
        std::shared_ptr<numeric::IrregularMesh> mesh;
        std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor;
        std::shared_ptr<numeric::SparseMatrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::Matrix<unsigned>> connectivityMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> sMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> rMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalSMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalRMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalJacobianMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> excitationMatrix;
        std::shared_ptr<numeric::Matrix<unsigned>> meshElements;
        dataType referenceValue;
    };

    namespace equation {
        // update matrix
        template <
            class dataType,
            bool logarithmic
        >
        void updateMatrix(const std::shared_ptr<numeric::Matrix<dataType>> elements,
            const std::shared_ptr<numeric::Matrix<dataType>> gamma,
            const std::shared_ptr<numeric::Matrix<unsigned>> connectivityMatrix, dataType referenceValue,
            cudaStream_t stream, std::shared_ptr<numeric::SparseMatrix<dataType>> matrix);
    }
}
}

#endif
