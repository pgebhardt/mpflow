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

#ifndef MPFLOW_INCLDUE_NUMERIC_BICGSTAB_H
#define MPFLOW_INCLDUE_NUMERIC_BICGSTAB_H

namespace mpFlow {
namespace numeric {
    // conjugate gradient class definition
    template <
        template <class type> class matrix_type
    >
    class BiCGSTAB {
    public:
        // constructor
        BiCGSTAB(dtype::size rows, dtype::size cols, cudaStream_t stream);

        // solve system
        void solve(const std::shared_ptr<matrix_type<dtype::real>> A,
            const std::shared_ptr<Matrix<dtype::real>> f, dtype::size iterations,
            bool dcFree, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> x);

        // member
        dtype::size rows;
        dtype::size cols;
        std::shared_ptr<Matrix<dtype::real>> r;
        std::shared_ptr<Matrix<dtype::real>> rHat;
        std::shared_ptr<Matrix<dtype::real>> roh;
        std::shared_ptr<Matrix<dtype::real>> rohOld;
        std::shared_ptr<Matrix<dtype::real>> alpha;
        std::shared_ptr<Matrix<dtype::real>> beta;
        std::shared_ptr<Matrix<dtype::real>> omega;
        std::shared_ptr<Matrix<dtype::real>> nu;
        std::shared_ptr<Matrix<dtype::real>> p;
        std::shared_ptr<Matrix<dtype::real>> t;
        std::shared_ptr<Matrix<dtype::real>> s;
        std::shared_ptr<Matrix<dtype::real>> temp1;
        std::shared_ptr<Matrix<dtype::real>> temp2;
    };

    // helper functions
    namespace bicgstab {
        void updateVector(const std::shared_ptr<Matrix<dtype::real>> x1,
            dtype::real sign, const std::shared_ptr<Matrix<dtype::real>> x2,
            const std::shared_ptr<Matrix<dtype::real>> scalar, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> result);
    }
}
}

#endif
