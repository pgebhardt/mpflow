// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SPARSE_MATRIX_H
#define FASTEIT_INCLUDE_SPARSE_MATRIX_H

// namespace fastEIT
namespace fastEIT {
    // sparse matrix class definition
    class SparseMatrix {
    public:
        // constructor and destructor
        SparseMatrix(dtype::size rows, dtype::size columns, cudaStream_t stream) {
            this->init(rows, columns, stream);
        }

        SparseMatrix(const std::shared_ptr<Matrix<dtype::real>> matrix, cudaStream_t stream);
        virtual ~SparseMatrix();

        // matrix multiply
        void multiply(const std::shared_ptr<Matrix<dtype::real>> matrix, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> result) const;

        // accessors
        dtype::size rows() const { return this->rows_; }
        dtype::size columns() const { return this->columns_; }
        dtype::size data_rows() const { return this->data_rows_; }
        dtype::size data_columns() const { return this->data_columns_; }
        dtype::size density() const { return this->density_; }
        const dtype::real* values() const { return this->values_; }
        const dtype::index* column_ids() const { return this->column_ids_; }

        // mutators:
        dtype::real* values() { return this->values_; }
        dtype::index* column_ids() { return this->column_ids_; }
        dtype::size& density() { return this->density_; }

    private:
        // init empty sparse matrix
        void init(dtype::size rows, dtype::size columns, cudaStream_t stream);

        // convert to sparse matrix
        void convert(const std::shared_ptr<Matrix<dtype::real>> matrix, cudaStream_t stream);

        // member
        dtype::size rows_;
        dtype::size columns_;
        dtype::size data_rows_;
        dtype::size data_columns_;
        dtype::size density_;
        dtype::real* values_;
        dtype::index* column_ids_;
    };
}

#endif
