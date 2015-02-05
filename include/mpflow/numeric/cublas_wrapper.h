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

    // dtype::real
    template <>
    class cublasWrapper<mpFlow::dtype::real> {
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

    // dtype::complex
    template <>
    class cublasWrapper<mpFlow::dtype::complex> {
    public:
        static cublasStatus_t copy(cublasHandle_t handle, int n, const dtype::complex* x,
            int incx, dtype::complex* y, int incy) {
            return cublasCcopy(handle, n, (const cuComplex*)x, incx, (cuComplex*)y, incy);
        }

        static cublasStatus_t axpy(cublasHandle_t handle, int n, const dtype::complex* alpha,
        const dtype::complex* x, int incx, dtype::complex* y, int incy) {
            return cublasCaxpy(handle, n, (const cuComplex*)alpha, (const cuComplex*)x,
                incx, (cuComplex*)y, incy);
        }

        static cublasStatus_t gemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
            int m, int n, int k, const dtype::complex* alpha, const dtype::complex* A, int lda,
            const dtype::complex* B, int ldb, const dtype::complex* beta, dtype::complex* C, int ldc) {
            return cublasCgemm(handle, transa, transb, m, n, k, (const cuComplex*)alpha, (const cuComplex*)A,
                lda, (const cuComplex*)B, ldb, (const cuComplex*)beta, (cuComplex*)C, ldc);
        }

        static cublasStatus_t gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
            const dtype::complex* alpha, const dtype::complex* A, int lda, const dtype::complex* x,
            int incx, const dtype::complex* beta, dtype::complex* y, int incy) {
            return cublasCgemv(handle, trans, m, n, (const cuComplex*)alpha, (const cuComplex*)A,
                lda, (const cuComplex*)x, incx, (const cuComplex*)beta, (cuComplex*)y, incy);
        }
    };
}
}

#endif
