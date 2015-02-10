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
        class dataType,
        class basisFunctionType_
    >
    class Equation {
    public:
        typedef basisFunctionType_ basisFunctionType;

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
            dtype::size driveCount, dtype::size measurmentCount, bool additiv,
            cudaStream_t stream, std::shared_ptr<numeric::Matrix<dataType>> result);

        void update(const std::shared_ptr<numeric::Matrix<dataType>> alpha,
            const dataType k, const std::shared_ptr<numeric::Matrix<dataType>> beta,
            cudaStream_t stream);

        // member
        std::shared_ptr<numeric::IrregularMesh> mesh;
        std::shared_ptr<FEM::BoundaryDescriptor> boundaryDescriptor;
        std::shared_ptr<numeric::SparseMatrix<dataType>> systemMatrix;
        std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> sMatrix;
        std::shared_ptr<numeric::SparseMatrix<dataType>> rMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalSMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> elementalRMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> elementalJacobianMatrix;
        std::shared_ptr<numeric::Matrix<dataType>> excitationMatrix;
        dataType referenceValue;
    };

    namespace equation {
        // reduce matrix
        template <
            class outputType,
            class inputType,
            class shapeType
        >
        std::shared_ptr<numeric::Matrix<outputType>> reduceMatrix(
            const std::vector<Eigen::Array<inputType, Eigen::Dynamic, Eigen::Dynamic>>& intermediateMatrices,
            const std::shared_ptr<numeric::SparseMatrix<shapeType>> shapeMatrix, cudaStream_t stream);

        // update matrix
        template <
            class dataType
        >
        void updateMatrix(const std::shared_ptr<numeric::Matrix<dataType>> elements,
            const std::shared_ptr<numeric::Matrix<dataType>> gamma,
            const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix, dataType referenceValue,
            cudaStream_t stream, std::shared_ptr<numeric::SparseMatrix<dataType>> matrix);
    }
}
}

#endif
