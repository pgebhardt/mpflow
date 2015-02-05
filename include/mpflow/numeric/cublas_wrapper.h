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
    class cublasWrapper<cuComplex> {
    public:
        template <typename... Args>
        static cublasStatus_t copy(Args&&... args) {
            return cublasCcopy(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t axpy(Args&&... args) {
            return cublasCaxpy(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t gemm(Args&&... args) {
            return cublasCgemm(std::forward<Args>(args)...);
        }

        template <typename... Args>
        static cublasStatus_t gemv(Args&&... args) {
            return cublasCgemv(std::forward<Args>(args)...);
        }
    };
}
}

#endif
