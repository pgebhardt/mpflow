// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MODEL_KERNEL_H
#define FASTEIT_INCLUDE_MODEL_KERNEL_H

// namespaces fastEIT
namespace fastEIT {
    // cuda kernel
    namespace modelKernel {
        // reduce connectivity and elemental matrices
        template <
            class type
        >
        void reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* intermediate_matrix, const dtype::index* column_ids,
            dtype::size rows, dtype::index offset, type* matrix);

        // update matrix kernel
        void updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::index* connectivity_matrix, const dtype::real* elemental_matrix,
            const dtype::real* gamma, dtype::real sigma_ref, dtype::size rows,
            dtype::size columns, dtype::real* matrix_values);
    }
}

#endif
