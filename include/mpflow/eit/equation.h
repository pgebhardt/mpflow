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

#ifndef MPFLOW_INCLDUE_EIT_EQUATION_H
#define MPFLOW_INCLDUE_EIT_EQUATION_H

namespace mpFlow {
namespace EIT {
    // model class describing an elliptical differential equation
    template <
        class _basisFunctionType
    >
    class Equation {
    public:
        typedef _basisFunctionType basisFunctionType;

        // class methods
        // update matrix
        static void updateMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> elements,
            const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
            const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix, dtype::real sigmaRef,
            cudaStream_t stream, std::shared_ptr<numeric::SparseMatrix<dtype::real>> matrix);

        // reduce matrix
        template <
            class type
        >
        static void reduceMatrix(const std::shared_ptr<numeric::Matrix<type>> intermediateMatrix,
            const std::shared_ptr<numeric::SparseMatrix<dtype::real>> shape, dtype::index offset,
            cudaStream_t stream, std::shared_ptr<numeric::Matrix<type>> matrix);

        // instance methods
        // constructor
        Equation(std::shared_ptr<numeric::IrregularMesh> mesh,
            std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor,
            dtype::real referenceValue, cudaStream_t stream);

        // init methods
        std::shared_ptr<numeric::Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);
        void initExcitationMatrix(cudaStream_t stream);
        void initJacobianCalculationMatrix(cudaStream_t stream);

        void calcJacobian(const std::shared_ptr<numeric::Matrix<dtype::real>> phi,
            const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
            dtype::size driveCount, dtype::size measurmentCount, bool additiv,
            cudaStream_t stream, std::shared_ptr<numeric::Matrix<dtype::real>> result);

        void update(const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
            dtype::real k, cudaStream_t stream);

        // member
        std::shared_ptr<numeric::IrregularMesh> mesh;
        std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> systemMatrix;
        std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> sMatrix;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> rMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> elementalSMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> elementalRMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> elementalJacobianMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> excitationMatrix;
        dtype::real referenceValue;
    };
}
}

#endif
