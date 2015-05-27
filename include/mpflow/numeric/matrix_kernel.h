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

#ifndef MPFLOW_INCLUDE_NUMERIC_MATRIX_KERNEL_H
#define MPFLOW_INCLUDE_NUMERIC_MATRIX_KERNEL_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // matrix kernel
    namespace matrixKernel {
        // fill kernel
        template <class type>
        void fill(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
            type const value, unsigned const rows, unsigned const cols,
            unsigned const dataRows, type* const result);

        // fill unity matrix
        template <class type>
        void setEye(dim3 const blocks, dim3 const threads, cudaStream_t const stream,
            unsigned const rows, unsigned const dataRows, type* matrix);
            
        // add kernel
        template <
            class type
        >
        void add(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* matrix, unsigned rows, type* result);

        // scale kernel
        template <
            class type
        >
        void scale(dim3 blocks, dim3 threads, cudaStream_t stream,
            type scalar, unsigned rows, type* result);

        // elementwise multiply kernel
        template <
            class type
        >
        void elementwiseMultiply(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* a, const type* b, unsigned rows, type* result);

        // elementwise division kernel
        template <
            class type
        >
        void elementwiseDivision(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* a, const type* b, unsigned rows, type* result);

        // vectorDotProduct kernel
        template <
            class type
        >
        void vectorDotProduct(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* a, const type* b, unsigned rows, type* result);

        // sum kernel
        template <
            class type
        >
        void sum(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* vector, unsigned rows, unsigned offset, type* result);

        // min kernel
        template <
            class type
        >
        void min(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* vector, unsigned rows, unsigned offset, type* result);

        // max kernel
        template <
            class type
        >
        void max(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* vector, unsigned rows, unsigned offset, type* result);
    }
}
}

#endif
