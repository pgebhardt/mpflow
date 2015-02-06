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
namespace equation {
    // reduce matrix
    template <
        class dataType,
        class shapeDataType
    >
    void reduceMatrix(const std::shared_ptr<numeric::Matrix<dataType>> intermediateMatrix,
        const std::shared_ptr<numeric::SparseMatrix<shapeDataType>> shape, dtype::index offset,
        cudaStream_t stream, std::shared_ptr<numeric::Matrix<dataType>> matrix);

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
