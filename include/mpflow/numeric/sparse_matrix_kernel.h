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

#ifndef MPFLOW_INCLUDE_NUMERIC_SPARSE_MATRIX_KERNEL_H
#define MPFLOW_INCLUDE_NUMERIC_SPARSE_MATRIX_KERNEL_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // sparse matrix kernel
    namespace sparseMatrixKernel {
        // convert to sparse matrix kernel
        template <
            class type
        >
        void convert(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* matrix, dtype::size rows, dtype::size columns,
            type* values, dtype::index* columnIds,
            dtype::index* elementCount);

        // convert to matrix kernel
        template <
            class type
        >
        void convertToMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* values, const dtype::index* column_ids,
            dtype::size density, dtype::size rows, type* matrix);

        // sparse matrix multiply kernel
        template <
            class type
        >
        void multiply(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* values, const dtype::index* columnIds,
            const type* matrix, dtype::size result_rows, dtype::size matrix_rows,
            dtype::size columns, dtype::size density, type* result);
    }
}
}

#endif
