// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_NUMERIC_CONJUGATE_KERNEL_H
#define MPFLOW_INCLDUE_NUMERIC_CONJUGATE_KERNEL_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // namespace conjugate
    namespace conjugateKernel {
        // add scalar kernel
        void addScalar(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* scalar, dtype::size vector_rows,
            dtype::size rows, dtype::size columns, dtype::real* vector);

        // update vector kernel
        void updateVector(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* x1, const dtype::real sign, const dtype::real* x2,
            const dtype::real* r1, const dtype::real* r2, dtype::size rows,
            dtype::real* result);

        // gemv kernel
        void gemv(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* matrix, const dtype::real* vector,
            dtype::size rows, dtype::real* result);

        // row reduce kernel
        void reduceRow(dim3 blocks, dim3 threads, cudaStream_t stream,
            dtype::size rows, dtype::real* vector);
    }
}
}

#endif
