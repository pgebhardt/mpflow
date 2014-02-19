// mpFlow
//
// Copyright (C) 2014  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_UWB_MODEL_KERNEL_H
#define MPFLOW_INCLDUE_UWB_MODEL_KERNEL_H

// namespaces mpFlow::UWB::modelKernel
namespace mpFlow {
namespace UWB {
namespace modelKernel {
    // reduce connectivity and elemental matrices
    template <
        class type
    >
    void reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const type* intermediateMatrix, const dtype::index* columnIds,
        dtype::size rows, dtype::index offset, type* matrix);

    // update matrix kernel
    void updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dtype::index* connectivityMatrix, const dtype::real* elementalMatrix,
        const dtype::real* material, dtype::size rows, dtype::size columns,
        dtype::real* matrixValues);

    // update system matrix kernel
    void updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const mpFlow::dtype::real* sMatrixValues, const mpFlow::dtype::real* rMatrixValues,
        const mpFlow::dtype::index* sMatrixColumnIds, mpFlow::dtype::size density,
        mpFlow::dtype::real sScalar, mpFlow::dtype::real rScalar,
        mpFlow::dtype::index rowOffset, mpFlow::dtype::index columnOffset,
        mpFlow::dtype::real* systemMatrixValues);
}
}
}

#endif
