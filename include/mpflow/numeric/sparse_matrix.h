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
        SparseMatrix(unsigned rows, unsigned cols, cudaStream_t stream) {
            this->init(rows, cols, stream);
        }

        SparseMatrix(const std::shared_ptr<Matrix<type>> matrix, cudaStream_t stream);
        virtual ~SparseMatrix();

        // copy methods
        void copy(const std::shared_ptr<SparseMatrix<type>> other, cudaStream_t stream=nullptr);
        void copyToDevice(cudaStream_t stream=nullptr);
        void copyToHost(cudaStream_t stream=nullptr);

        // convert to matrix
        std::shared_ptr<Matrix<type>> toMatrix(cudaStream_t stream);

        // matrix multiply
        void multiply(const std::shared_ptr<Matrix<type>> matrix, cudaStream_t stream,
            std::shared_ptr<Matrix<type>> result) const;

        // accessors
        unsigned getColumnId(unsigned row, unsigned col) const;
        type getValue(unsigned row, unsigned col) const;
        void setValue(unsigned row, unsigned col, const type& value);

        // member
        unsigned rows;
        unsigned cols;
        unsigned dataRows;
        unsigned dataCols;
        unsigned density;
        type* deviceValues;
        type* hostValues;
        unsigned* deviceColumnIds;
        unsigned* hostColumnIds;

    private:
        // init empty sparse matrix
        void init(unsigned rows, unsigned cols, cudaStream_t stream);

        // convert to sparse matrix
        void convert(const std::shared_ptr<Matrix<type>> matrix, cudaStream_t stream);

    };
}
}

#endif
