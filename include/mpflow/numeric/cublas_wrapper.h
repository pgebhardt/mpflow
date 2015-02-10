// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLUDE_NUMERIC_CUBLAS_WRAPPER_H
#define MPFLOW_INCLUDE_NUMERIC_CUBLAS_WRAPPER_H

namespace mpFlow {
namespace numeric {
    template <
        class dataType
    >
    class cublasWrapper {
    public:
        template <typename... Args>
        static cublasStatus_t copy(Args&&...) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }

        template <typename... Args>
        static cublasStatus_t axpy(Args&&...) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }

        template <typename... Args>
        static cublasStatus_t gemm(Args&&...) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }

        template <typename... Args>
        static cublasStatus_t gemv(Args&&...) {
            return CUBLAS_STATUS_NOT_SUPPORTED;
        }
    };

    // float
    template <>
    class cublasWrapper<float> {
    public:
        template <typename... Args>
        static cublasStatus_t copy(Args&&... args) {
            return cublasScopy(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t axpy(Args&&... args) {
            return cublasSaxpy(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t gemm(Args&&... args) {
            return cublasSgemm(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t gemv(Args&&... args) {
            return cublasSgemv(std::forward<Args>(args)...);
        }
    };

    // double
    template <>
    class cublasWrapper<double> {
    public:
        template <typename... Args>
        static cublasStatus_t copy(Args&&... args) {
            return cublasDcopy(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t axpy(Args&&... args) {
            return cublasDaxpy(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t gemm(Args&&... args) {
            return cublasDgemm(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t gemv(Args&&... args) {
            return cublasDgemv(std::forward<Args>(args)...);
        }
    };

    // thrust::complex<float>
    template <>
    class cublasWrapper<thrust::complex<float>> {
    public:
        static cublasStatus_t copy(cublasHandle_t handle, int n, const thrust::complex<float>* x,
            int incx, thrust::complex<float>* y, int incy) {
            return cublasCcopy(handle, n, (const cuComplex*)x, incx, (cuComplex*)y, incy);
        }

        static cublasStatus_t axpy(cublasHandle_t handle, int n, const thrust::complex<float>* alpha,
        const thrust::complex<float>* x, int incx, thrust::complex<float>* y, int incy) {
            return cublasCaxpy(handle, n, (const cuComplex*)alpha, (const cuComplex*)x,
                incx, (cuComplex*)y, incy);
        }

        static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k, const thrust::complex<float>* alpha, const thrust::complex<float>* A, int lda,
            const thrust::complex<float>* B, int ldb, const thrust::complex<float>* beta, thrust::complex<float>* C, int ldc) {
            return cublasCgemm(handle, transa, transb, m, n, k, (const cuComplex*)alpha, (const cuComplex*)A,
                lda, (const cuComplex*)B, ldb, (const cuComplex*)beta, (cuComplex*)C, ldc);
        }

        static cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
            const thrust::complex<float>* alpha, const thrust::complex<float>* A, int lda, const thrust::complex<float>* x,
            int incx, const thrust::complex<float>* beta, thrust::complex<float>* y, int incy) {
            return cublasCgemv(handle, trans, m, n, (const cuComplex*)alpha, (const cuComplex*)A,
                lda, (const cuComplex*)x, incx, (const cuComplex*)beta, (cuComplex*)y, incy);
        }
    };

    // thrust::complex<double>
    template <>
    class cublasWrapper<thrust::complex<double>> {
    public:
        static cublasStatus_t copy(cublasHandle_t handle, int n, const thrust::complex<double>* x,
            int incx, thrust::complex<double>* y, int incy) {
            return cublasZcopy(handle, n, (const cuDoubleComplex*)x, incx, (cuDoubleComplex*)y, incy);
        }

        static cublasStatus_t axpy(cublasHandle_t handle, int n, const thrust::complex<double>* alpha,
        const thrust::complex<double>* x, int incx, thrust::complex<double>* y, int incy) {
            return cublasZaxpy(handle, n, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)x,
                incx, (cuDoubleComplex*)y, incy);
        }

        static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k, const thrust::complex<double>* alpha, const thrust::complex<double>* A, int lda,
            const thrust::complex<double>* B, int ldb, const thrust::complex<double>* beta, thrust::complex<double>* C, int ldc) {
            return cublasZgemm(handle, transa, transb, m, n, k, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A,
                lda, (const cuDoubleComplex*)B, ldb, (const cuDoubleComplex*)beta, (cuDoubleComplex*)C, ldc);
        }

        static cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
            const thrust::complex<double>* alpha, const thrust::complex<double>* A, int lda, const thrust::complex<double>* x,
            int incx, const thrust::complex<double>* beta, thrust::complex<double>* y, int incy) {
            return cublasZgemv(handle, trans, m, n, (const cuDoubleComplex*)alpha, (const cuDoubleComplex*)A,
                lda, (const cuDoubleComplex*)x, incx, (const cuDoubleComplex*)beta, (cuDoubleComplex*)y, incy);
        }
    };
}
}

#endif
