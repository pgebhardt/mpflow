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
        void convert(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* matrix, dtype::size rows, dtype::size columns,
            dtype::real* values, dtype::index* columnIds,
            dtype::index* elementCount);

        // convert to matrix kernel
        void convertToMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* values, const dtype::index* column_ids,
            dtype::size density, dtype::size rows, dtype::real* matrix);

        // sparse matrix multiply kernel
        void multiply(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* values, const dtype::index* columnIds,
            const dtype::real* matrix, dtype::size result_rows, dtype::size matrix_rows,
            dtype::size columns, dtype::size density, dtype::real* result);
    }
}

#endif
