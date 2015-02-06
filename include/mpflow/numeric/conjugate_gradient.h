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
        class dataType,
        template <class> class matrixType
    >
    class ConjugateGradient {
    public:
        // constructor
        ConjugateGradient(dtype::size rows, dtype::size columns, cudaStream_t stream);

        // solve system
        void solve(const std::shared_ptr<matrixType<dataType>> A,
            const std::shared_ptr<Matrix<dataType>> f, dtype::size iterations,
            cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dataType>> x,
            dtype::real tolerance=0.0, bool dcFree=false);

        // member
        dtype::size rows;
        dtype::size cols;
        std::shared_ptr<Matrix<dataType>> r;
        std::shared_ptr<Matrix<dataType>> p;
        std::shared_ptr<Matrix<dataType>> roh;
        std::shared_ptr<Matrix<dataType>> rohOld;
        std::shared_ptr<Matrix<dataType>> temp1;
        std::shared_ptr<Matrix<dataType>> temp2;
    };

    // helper functions
    namespace conjugateGradient {
        template <
            class dataType
        >
        void addScalar(const std::shared_ptr<Matrix<dataType>> scalar,
            dtype::size rows, dtype::size columns, cudaStream_t stream,
            std::shared_ptr<Matrix<dataType>> vector);

        template <
            class dataType
        >
        void updateVector(const std::shared_ptr<Matrix<dataType>> x1,
            dtype::real sign, const std::shared_ptr<Matrix<dataType>> x2,
            const std::shared_ptr<Matrix<dataType>> r1,
            const std::shared_ptr<Matrix<dataType>> r2, cudaStream_t stream,
            std::shared_ptr<Matrix<dataType>> result);
    }
}
}

#endif
