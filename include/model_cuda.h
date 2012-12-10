// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MODEL_CUDA_H
#define FASTEIT_INCLUDE_MODEL_CUDA_H

// namespaces fastEIT
namespace fastEIT {
    // cuda kernel
    namespace model {
        // update matrix
        void updateMatrix(const Matrix<dtype::real>* elements, const Matrix<dtype::real>* gamma,
            const Matrix<dtype::index>* connectivityMatrix, dtype::real sigmaRef, cudaStream_t stream,
            SparseMatrix* matrix);

        // reduce matrix
        template <
            class type
        >
        void reduceMatrix(const Matrix<type>* intermediateMatrix, const SparseMatrix* shape,
            cudaStream_t stream, Matrix<type>* matrix);
    }
}

#endif
