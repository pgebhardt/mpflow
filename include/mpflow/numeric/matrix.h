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

#ifndef MPFLOW_INCLUDE_NUMERIC_MATRIX_H
#define MPFLOW_INCLUDE_NUMERIC_MATRIX_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // forward declarations
    template <class type> class SparseMatrix;

    // matrix class definition
    template <
        class type = dtype::real
    >
    class Matrix {
    public:
        // constructor and destructor
        Matrix(dtype::size rows, dtype::size cols, cudaStream_t stream,
            type value=0);
        virtual ~Matrix();
        void fill(type value, cudaStream_t stream);

        // copy methods
        void copy(const std::shared_ptr<Matrix<type>> other, cudaStream_t stream);
        void copyToDevice(cudaStream_t stream);
        void copyToHost(cudaStream_t stream);

        // mathematical methods
        void add(const std::shared_ptr<Matrix<type>> value, cudaStream_t stream);
        void multiply(const std::shared_ptr<Matrix<type>> A,
            const std::shared_ptr<Matrix<type>> B, cublasHandle_t handle,
            cudaStream_t stream);
        void multiply(const std::shared_ptr<SparseMatrix<type>> A,
            const std::shared_ptr<Matrix<type>> B, cublasHandle_t handle,
            cudaStream_t stream);
        void scalarMultiply(type scalar, cudaStream_t stream);
        void vectorDotProduct(const std::shared_ptr<Matrix<type>> A,
            const std::shared_ptr<Matrix<type>> B, cudaStream_t stream);

        // reduce methods
        void sum(const std::shared_ptr<Matrix<type>> value, cudaStream_t stream);
        void min(const std::shared_ptr<Matrix<type>> value, cudaStream_t stream);
        void max(const std::shared_ptr<Matrix<type>> value, cudaStream_t stream);

        // accessors
        const type& operator() (dtype::index i, dtype::index j) const {
            // check index
            if ((i >= this->rows) || (j >= this->cols)) {
                throw std::invalid_argument("mpFlow::numeric::Matrix::operator(): index out of range");
            }

            return this->hostData[i + j * this->dataRows];
        }

        // mutators
        type& operator() (dtype::index i, dtype::index j) {
            // check index
            if ((i >= this->rows) || (j >= this->cols)) {
                throw std::invalid_argument("mpFlow::numeric::Matrix::operator(): index out of range");
            }

            return this->hostData[i + j * this->dataRows];
        }

        type* hostData;
        type* deviceData;
        dtype::size rows;
        dtype::size cols;
        dtype::size dataRows;
        dtype::size dataCols;
    };

    // namespace matrix
    namespace matrix {
        // load matrix from stream
        template <
            class type
        >
        std::shared_ptr<mpFlow::numeric::Matrix<type>> loadtxt(std::istream* istream,
            cudaStream_t stream);

        // load matrix from file
        template <
            class type
        >
        std::shared_ptr<mpFlow::numeric::Matrix<type>> loadtxt(const std::string filename,
            cudaStream_t stream);

        // save matrix to stream
        template <
            class type
        >
        void savetxt(const std::shared_ptr<Matrix<type>> matrix, std::ostream* ostream);

        // save matrix to file
        template <
            class type
        >
        void savetxt(const std::string filename, const std::shared_ptr<Matrix<type>> matrix);

        // converts matrix to eigen array
        template <
            class type
        >
        Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> toEigen(
            std::shared_ptr<Matrix<type>> matrix);

        // converts eigen array to matrix
        template <
            class mpflow_type,
            class eigen_type
        >
        std::shared_ptr<Matrix<mpflow_type>> fromEigen(
            const Eigen::Ref<Eigen::Array<eigen_type, Eigen::Dynamic, Eigen::Dynamic>>& array,
            cudaStream_t stream);
    }
}
}

#endif
