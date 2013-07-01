// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_NUMERIC_MATRIX_KERNEL_H
#define MPFLOW_INCLUDE_NUMERIC_MATRIX_KERNEL_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // matrix kernel
    namespace matrixKernel {
        // add kernel
        template <
            class type
        >
        void add(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* matrix, dtype::size rows, type* result);

        // scale kernel
        template <
            class type
        >
        void scale(dim3 blocks, dim3 threads, cudaStream_t stream,
            type scalar, dtype::size rows, type* result);

        // vector dot product kernel
        template <
            class type
        >
        void vectorDotProduct(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* a, const type* b, dtype::size rows, type* result);

        // sum kernel
        template <
            class type
        >
        void sum(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* vector, dtype::size rows, dtype::size offset, type* result);

        // min kernel
        template <
            class type
        >
        void min(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* vector, dtype::size rows, dtype::size offset, type* result);

        // max kernel
        template <
            class type
        >
        void max(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* vector, dtype::size rows, dtype::size offset, type* result);
    }
}
}

#endif
