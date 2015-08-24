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
// Copyright (C) 2015 Patrik Gebhardt
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
        class type_
    >
    class Matrix {
    public:
        typedef type_ type;

        // constructor and destructor
        Matrix(unsigned const rows, unsigned const cols, cudaStream_t const stream=nullptr,
            type const value=0, bool const allocateHostMemory=true);
        virtual ~Matrix();

        // create special matrices
        static std::shared_ptr<Matrix<type>> eye(unsigned const size,
            cudaStream_t const stream=nullptr);

        // copy methods
        void copy(std::shared_ptr<Matrix<type> const> const other, cudaStream_t const stream=nullptr);
        void copyToDevice(cudaStream_t const stream=nullptr);
        void copyToHost(cudaStream_t const stream=nullptr) const;

        // mathematical methods
        void fill(type const value, cudaStream_t const stream=nullptr);
        void setEye(cudaStream_t const stream=nullptr);
        void diag(std::shared_ptr<Matrix<type> const> const matrix, cudaStream_t const stream=nullptr);        
        void add(type const value, cudaStream_t const stream=nullptr);
        void add(std::shared_ptr<Matrix<type> const> const value, cudaStream_t const stream=nullptr);
        void multiply(std::shared_ptr<Matrix<type> const> const A,
            std::shared_ptr<Matrix<type> const> const B, cublasHandle_t const handle,
            cudaStream_t const stream=nullptr, cublasOperation_t const transA=CUBLAS_OP_N,
            cublasOperation_t const transB=CUBLAS_OP_N);
        void multiply(std::shared_ptr<SparseMatrix<type> const> const A,
            std::shared_ptr<Matrix<type> const> const B, cublasHandle_t const handle,
            cudaStream_t const stream=nullptr);
        void scalarMultiply(type const scalar, cudaStream_t const stream=nullptr);
        void elementwiseMultiply(std::shared_ptr<Matrix<type> const> const A,
            std::shared_ptr<Matrix<type> const> const B, cudaStream_t const stream=nullptr);
        void elementwiseDivision(std::shared_ptr<Matrix<type> const> const A,
            std::shared_ptr<Matrix<type> const> const B, cudaStream_t const stream=nullptr);
        void vectorDotProduct(std::shared_ptr<Matrix<type> const> const A,
            std::shared_ptr<Matrix<type> const> const B, cudaStream_t const stream=nullptr);
        void setIndexedElements(std::shared_ptr<Matrix<unsigned> const> const indices,
            type const value, cudaStream_t const stream=nullptr);
            
        // reduce methods
        void sum(std::shared_ptr<Matrix<type> const> const value, cudaStream_t const stream=nullptr);
        void min(std::shared_ptr<Matrix<type> const> const value, cudaStream_t const stream=nullptr);
        void max(std::shared_ptr<Matrix<type> const> const value, cudaStream_t const stream=nullptr);

        // save matrix as txt
        void savetxt(std::ostream& ostream, char const delimiter=' ') const;
        void savetxt(std::string const filename, char const delimiter=' ') const;

        // load txt formatted matrix from file
        static std::shared_ptr<mpFlow::numeric::Matrix<type>> loadtxt(std::istream& istream,
            cudaStream_t const stream=nullptr, char const delimiter=' ');
        static std::shared_ptr<mpFlow::numeric::Matrix<type>> loadtxt(std::string const filename,
            cudaStream_t const stream=nullptr, char const delimiter=' ');

        // cast from and to eigen arrays
        Eigen::Array<typename typeTraits::convertComplexType<type>::type,
            Eigen::Dynamic, Eigen::Dynamic> toEigen(cudaStream_t const stream=nullptr) const;
        static std::shared_ptr<mpFlow::numeric::Matrix<type>> fromEigen(
            Eigen::Ref<Eigen::Array<typename typeTraits::convertComplexType<type>::type,
                Eigen::Dynamic, Eigen::Dynamic> const> const array,
            cudaStream_t const stream=nullptr);

        // create from json array
#ifdef _JSON_H
        static std::shared_ptr<Matrix<type>> fromJsonArray(json_value const& array,
            cudaStream_t const stream=nullptr);
#endif

        // I/O operators
        friend std::ostream& operator << (std::ostream& out, Matrix<type> const& matrix) {
            matrix.savetxt(out);
            return out;
        }

        // accessors
        const type operator() (unsigned const i, unsigned const j) const {
            // check index
            if (this->hostData == nullptr) {
                throw std::logic_error("mpFlow::numeric::Matrix::operator(): host memory was not allocated");
            }
            if ((i >= this->rows) || (j >= this->cols)) {
                throw std::invalid_argument(
                    str::format("mpFlow::numeric::Matrix::operator(): index out of range: (%d, %d) >= (%d, %d)")
                        (i, j, this->rows, this->cols));
            }

            return this->hostData[i + j * this->dataRows];
        }

        // mutators
        type& operator() (unsigned const i, unsigned const j) {
            // check index
            if (this->hostData == nullptr) {
                throw std::logic_error("mpFlow::numeric::Matrix::operator(): host memory was not allocated");
            }
            if ((i >= this->rows) || (j >= this->cols)) {
                throw std::invalid_argument(
                    str::format("mpFlow::numeric::Matrix::operator(): index out of range: (%d, %d) >= (%d, %d)")
                        (i, j, this->rows, this->cols));
            }

            return this->hostData[i + j * this->dataRows];
        }

        type* hostData;
        type* deviceData;
        unsigned const rows;
        unsigned const cols;
        unsigned dataRows;
        unsigned dataCols;
    };

#ifdef _JSON_H
    // creates eigen array from an json array
    template <class type>
    Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> eigenFromJsonArray(
        json_value const& array) {
        // check type of json value
        if (array.type != json_array) {
            return Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic>();
        }

        // exctract sizes
        unsigned const rows = array.u.array.length;
        unsigned const cols = array[0].type == json_array ? array[0].u.array.length : 1;

        // create array
        Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> eigenArray(rows, cols);

        // extract values
        if (array[0].type != json_array) {
            for (unsigned row = 0; row < eigenArray.rows(); ++row) {
                eigenArray(row, 0) = array[row].u.dbl;
            }
        }
        else {
            for (unsigned row = 0; row < eigenArray.rows(); ++row)
            for (unsigned col = 0; col < eigenArray.cols(); ++col) {
                eigenArray(row, col) = array[row][col].u.dbl;
            }
        }

        return eigenArray;
    }
#endif
}
}

#endif
