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
        class dataType,
        template <class> class matrixType
    >
    class BiCGSTAB {
    public:
        // constructor
        BiCGSTAB(dtype::size rows, dtype::size cols, cudaStream_t stream);

        // solve system
        dtype::index solve(const std::shared_ptr<matrixType<dataType>> A,
            const std::shared_ptr<Matrix<dataType>> f, dtype::size iterations,
            cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dataType>> x,
            dtype::real tolerance=0.0, bool dcFree=false);

        // member
        dtype::size rows;
        dtype::size cols;
        std::shared_ptr<Matrix<dataType>> r;
        std::shared_ptr<Matrix<dataType>> rHat;
        std::shared_ptr<Matrix<dataType>> roh;
        std::shared_ptr<Matrix<dataType>> rohOld;
        std::shared_ptr<Matrix<dataType>> alpha;
        std::shared_ptr<Matrix<dataType>> beta;
        std::shared_ptr<Matrix<dataType>> omega;
        std::shared_ptr<Matrix<dataType>> nu;
        std::shared_ptr<Matrix<dataType>> p;
        std::shared_ptr<Matrix<dataType>> t;
        std::shared_ptr<Matrix<dataType>> s;
        std::shared_ptr<Matrix<dataType>> error;
        std::shared_ptr<Matrix<dataType>> temp1;
        std::shared_ptr<Matrix<dataType>> temp2;
    };

    // helper functions
    namespace bicgstab {
        template <
            class dataType
        >
        void updateVector(const std::shared_ptr<Matrix<dataType>> x1,
            dtype::real sign, const std::shared_ptr<Matrix<dataType>> x2,
            const std::shared_ptr<Matrix<dataType>> scalar, cudaStream_t stream,
            std::shared_ptr<Matrix<dataType>> result);
    }
}
}

#endif
