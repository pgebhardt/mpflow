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
    // reduce connectivity and elemental matrices
    template <
        class inputType,
        class outputType
    >
    void reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const inputType* intermediate_matrix, const dtype::index* column_ids,
        dtype::size rows, dtype::index offset, outputType* matrix);

    // update matrix kernel
    template <
        class dataType
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
        int nodesPerElement
    >
    void calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dataType* drive_phi, const dataType* measurment_phi,
        const dtype::index* connectivity_matrix,
        const dtype::real* elemental_jacobian_matrix, const dataType* factor,
        dataType referenceValue, dtype::size rows, dtype::size columns,
        dtype::size phi_rows, dtype::size element_count, dtype::size drive_count,
        dtype::size measurment_count, bool additiv, dataType* jacobian);
}
}
}

#endif
