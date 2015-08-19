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

#ifndef MPFLOW_INCLDUE_NUMERIC_CPU_SOLVER_H
#define MPFLOW_INCLDUE_NUMERIC_CPU_SOLVER_H

namespace mpFlow {
namespace numeric {
    // solves system of linear equations on CPU using Eigen
    template <
        class dataType
    >
    class CPUSolver {
    public:
        // constructor
        CPUSolver(unsigned const, unsigned const, cudaStream_t const) {}

        // solve system
        template <
            template <class> class matrixType,
            template <class> class preconditionerType = matrixType
        >
        unsigned solve(std::shared_ptr<matrixType<dataType>> const A,
            std::shared_ptr<Matrix<dataType> const> const b, cublasHandle_t const handle,
            cudaStream_t const stream, std::shared_ptr<Matrix<dataType>> const x,
            std::shared_ptr<preconditionerType<dataType>> const KInv=nullptr, unsigned const maxIterations=0,
            bool const dcFree=false);
    };
}
}

#endif
