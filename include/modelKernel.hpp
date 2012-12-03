// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MODEL_KERNEL_HPP
#define FASTEIT_INCLUDE_MODEL_KERNEL_HPP

// namespaces fastEIT
namespace fastEIT {
    // cuda kernel
    namespace modelKernel {
        // update matrix
        void updateMatrix(const Matrix<dtype::real>& elements, const Matrix<dtype::real>& gamma,
            const Matrix<dtype::index>& connectivityMatrix, dtype::real sigmaRef, cudaStream_t stream,
            SparseMatrix& matrix);

        // reduce matrix
        void reduceMatrix(const Matrix<dtype::real>& intermediateMatrix, const SparseMatrix& shape,
            cudaStream_t stream, Matrix<dtype::real>& matrix);
        void reduceMatrix(const Matrix<dtype::index>& intermediateMatrix, const SparseMatrix& shape,
            cudaStream_t stream, Matrix<dtype::index>& matrix);
    }
}

#endif
