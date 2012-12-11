// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MATRIX_CUDA_H
#define FASTEIT_INCLUDE_MATRIX_CUDA_H

// namespace fastEIT
namespace fastEIT {
    // matrix kernel
    namespace matrixKernel {
        // add kernel
        template <
            class type
        >
        void add(dim3 blocks, dim3 threads, cudaStream_t stream,
            const type* matrix, fastEIT::dtype::size rows, type* result);
    }
}

#endif
