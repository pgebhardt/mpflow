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

#ifndef MPFLOW_INCLUDE_NUMERIC_SPARSE_MATRIX_H
#define MPFLOW_INCLUDE_NUMERIC_SPARSE_MATRIX_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // sparse matrix class definition
    template <
        class type_
    >
    class SparseMatrix {
    public:
        typedef type_ type;

        // constructor and destructor
        SparseMatrix(unsigned const rows, unsigned const cols, cudaStream_t const stream)
            : rows(rows), cols(cols) {
            this->init(rows, cols, stream);
        }

        SparseMatrix(std::shared_ptr<Matrix<type> const> const matrix, cudaStream_t const stream);
        virtual ~SparseMatrix();

        // copy methods
        void copy(std::shared_ptr<SparseMatrix<type> const> const other, cudaStream_t const stream=nullptr);
        void copyToDevice(cudaStream_t const stream=nullptr);
        void copyToHost(cudaStream_t const stream=nullptr);

        // convert to matrix
        std::shared_ptr<Matrix<type>> toMatrix(cudaStream_t const stream) const;

        // arithmetic operations
        void scalarMultiply(type const scalar, cudaStream_t const stream);
        void multiply(std::shared_ptr<Matrix<type> const> const matrix, cudaStream_t const stream,
            std::shared_ptr<Matrix<type>> result) const;

        // cast to eigen array
        Eigen::Array<typename typeTraits::convertComplexType<type>::type,
            Eigen::Dynamic, Eigen::Dynamic> toEigen() const;

        // save matrix as txt
        void savetxt(std::ostream& ostream, char const delimiter=' ') const;
        void savetxt(std::string const filename, char const delimiter=' ') const;

        // accessors
        unsigned getColumnId(unsigned const row, unsigned const col) const;
        type getValue(unsigned const row, unsigned const col) const;
        void setValue(unsigned const row, unsigned const col, type const& value);

        // member
        unsigned const rows;
        unsigned const cols;
        unsigned dataRows;
        unsigned dataCols;
        unsigned density;
        type* deviceValues;
        type* hostValues;
        unsigned* deviceColumnIds;
        unsigned* hostColumnIds;

    private:
        // init empty sparse matrix
        void init(unsigned const rows, unsigned const cols, cudaStream_t const stream);

        // convert to sparse matrix
        void convert(std::shared_ptr<Matrix<type> const> const matrix, cudaStream_t const stream);
    };
}
}

#endif
