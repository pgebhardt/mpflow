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
        class type = mpFlow::dtype::real
    >
    class SparseMatrix {
    public:
        // constructor and destructor
        SparseMatrix(dtype::size rows, dtype::size columns, cudaStream_t) {
            this->init(rows, columns);
        }

        SparseMatrix(const std::shared_ptr<Matrix<type>> matrix, cudaStream_t stream);
        virtual ~SparseMatrix();

        // convert to matrix
        std::shared_ptr<Matrix<type>> toMatrix(cudaStream_t stream);

        // matrix multiply
        void multiply(const std::shared_ptr<Matrix<type>> matrix, cudaStream_t stream,
            std::shared_ptr<Matrix<type>> result) const;

        // accessors
        dtype::size rows() const { return this->rows_; }
        dtype::size columns() const { return this->columns_; }
        dtype::size data_rows() const { return this->data_rows_; }
        dtype::size data_columns() const { return this->data_columns_; }
        dtype::size density() const { return this->density_; }
        const type* values() const { return this->values_; }
        const dtype::index* column_ids() const { return this->column_ids_; }

        // mutators:
        type* values() { return this->values_; }
        dtype::index* column_ids() { return this->column_ids_; }
        dtype::size& density() { return this->density_; }

    private:
        // init empty sparse matrix
        void init(dtype::size rows, dtype::size columns);

        // convert to sparse matrix
        void convert(const std::shared_ptr<Matrix<type>> matrix, cudaStream_t stream);

        // member
        dtype::size rows_;
        dtype::size columns_;
        dtype::size data_rows_;
        dtype::size data_columns_;
        dtype::size density_;
        type* values_;
        dtype::index* column_ids_;
    };
}
}

#endif
