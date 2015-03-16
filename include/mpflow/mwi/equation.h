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

#ifndef MPFLOW_INCLDUE_MWI_EQUATION_H
#define MPFLOW_INCLDUE_MWI_EQUATION_H

namespace mpFlow {
namespace MWI {
    // model class describing an 2D TE Helmholz equation
    class Equation {
    public:
        // constructor
        Equation(std::shared_ptr<numeric::IrregularMesh> mesh,
            cudaStream_t stream);

        // init methods
        void initElementalMatrices(cudaStream_t stream);
        void initJacobianCalculationMatrix(cudaStream_t stream);

        void update(const std::shared_ptr<numeric::Matrix<thrust::complex<float>>> beta,
            thrust::complex<float> k, cudaStream_t stream);

        // member
        std::shared_ptr<numeric::IrregularMesh> mesh;
        std::shared_ptr<numeric::SparseMatrix<thrust::complex<float>>> systemMatrix;
        std::shared_ptr<numeric::Matrix<unsigned>> connectivityMatrix;
        std::shared_ptr<numeric::SparseMatrix<thrust::complex<float>>> sMatrix;
        std::shared_ptr<numeric::SparseMatrix<thrust::complex<float>>> rMatrix;
        std::shared_ptr<numeric::Matrix<thrust::complex<float>>> elementalSMatrix;
        std::shared_ptr<numeric::Matrix<thrust::complex<float>>> elementalRMatrix;
        std::shared_ptr<numeric::Matrix<float>> elementalJacobianMatrix;
    };
}
}

#endif
