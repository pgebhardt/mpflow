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

#ifndef MPFLOW_INCLDUE_FEM_EQUATION_KERNEL_H
#define MPFLOW_INCLDUE_FEM_EQUATION_KERNEL_H

namespace mpFlow {
namespace FEM {
namespace equationKernel {
    // update matrix kernel
    template <
        class dataType,
        bool logarithmic
    >
    void updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dtype::index* connectivity_matrix, const dataType* elemental_matrix,
        const dataType* gamma, dataType referenceValue, dtype::size rows,
        dtype::size columns, dataType* matrix_values);

    // update system matrix kernel
    template <
        class dataType
    >
    void updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dataType* sMatrixValues, const dataType* rMatrixValues,
        const dtype::index* sMatrixColumnIds, dtype::size density, dataType k,
        dataType* systemMatrixValues);

    // calc jacobian kernel
    template <
        class dataType,
        int nodesPerElement,
        bool logarithmic
    >
    void calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dataType* drivePhi, const dataType* measurmentPhi,
        const dtype::index* connectivityMatrix, const dataType* elementalJacobianMatrix,
        const dataType* gamma, dataType referenceValue, dtype::size rows,
        dtype::size columns, dtype::size phiRows, dtype::size elementCount,
        dtype::size driveCount, dtype::size measurmentCount, bool additiv, dataType* jacobian);
}
}
}

#endif
