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

#ifndef MPFLOW_INCLDUE_NUMERIC_CONJUGATE_GRADIENT_H
#define MPFLOW_INCLDUE_NUMERIC_CONJUGATE_GRADIENT_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // conjugate gradient class definition
    template <
        class dataType
    >
    class ConjugateGradient {
    public:
        // constructor
        ConjugateGradient(unsigned const rows, unsigned const columns, cudaStream_t const stream);

        // solve system
        template <
            template <class> class matrixType,
            template <class> class preconditionerType = matrixType
        >
        unsigned solve(std::shared_ptr<matrixType<dataType>> const A,
            std::shared_ptr<Matrix<dataType> const> const b, cublasHandle_t const handle,
            cudaStream_t const stream, std::shared_ptr<Matrix<dataType>> const x,
            std::shared_ptr<preconditionerType<dataType>> const KInv=nullptr, unsigned const maxIterations=0);

        // helper
        static void addScalar(std::shared_ptr<Matrix<dataType> const> const scalar,
            unsigned const rows, unsigned const columns, cudaStream_t const stream,
            std::shared_ptr<Matrix<dataType>> const vector);
        static void updateVector(std::shared_ptr<Matrix<dataType> const> const x1,
            double const sign, std::shared_ptr<Matrix<dataType> const> const x2,
            std::shared_ptr<Matrix<dataType> const> const r1,
            std::shared_ptr<Matrix<dataType> const> const r2, cudaStream_t const stream,
            std::shared_ptr<Matrix<dataType>> const result);

        // member
        unsigned const rows;
        unsigned const cols;

    private:
        std::shared_ptr<Matrix<dataType>> r;
        std::shared_ptr<Matrix<dataType>> p;
        std::shared_ptr<Matrix<dataType>> z;
        std::shared_ptr<Matrix<dataType>> roh;
        std::shared_ptr<Matrix<dataType>> rohOld;
        std::shared_ptr<Matrix<dataType>> reference;
        std::shared_ptr<Matrix<dataType>> temp1;
        std::shared_ptr<Matrix<dataType>> temp2;
    };
}
}

#endif
