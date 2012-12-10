// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_CONJUGATE_CUDA_H
#define FASTEIT_INCLUDE_CONJUGATE_CUDA_H

// namespace fastEIT
namespace fastEIT {
    // namespace numeric
    namespace numeric {
        // namespace conjugate
        namespace conjugate {
            void addScalar(const Matrix<dtype::real>* scalar, dtype::size rows,
                dtype::size columns, cudaStream_t stream, Matrix<dtype::real>* vector);

            void updateVector(const Matrix<dtype::real>* x1, dtype::real sign,
                const Matrix<dtype::real>* x2, const Matrix<dtype::real>* r1,
                const Matrix<dtype::real>* r2, cudaStream_t stream, Matrix<dtype::real>* result);

            void gemv(const Matrix<dtype::real>* matrix, const Matrix<dtype::real>* vector,
                cudaStream_t stream, Matrix<dtype::real>* result);

        }
    }
}

#endif
