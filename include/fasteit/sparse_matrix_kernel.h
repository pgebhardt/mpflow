// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SPARSE_MATRIX_KERNEL_H
#define FASTEIT_INCLUDE_SPARSE_MATRIX_KERNEL_H

// namespace fastEIT
namespace fastEIT {
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

#endif
