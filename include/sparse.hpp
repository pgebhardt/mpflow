// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SPARSE_HPP
#define FASTEIT_INCLUDE_SPARSE_HPP

// namespace fastEIT
namespace fastEIT {
    // sparse matrix class definition
    class SparseMatrix {
    public:
        // constructor and destructor
        SparseMatrix(dtype::size rows, dtype::size columns, cudaStream_t stream) {
            this->init(rows, columns, stream);
        }

        SparseMatrix(const Matrix<dtype::real>& matrix, cudaStream_t stream);
        virtual ~SparseMatrix();

    private:
        // init empty sparse matrix
        void init(dtype::size rows, dtype::size columns, cudaStream_t stream);

        // convert to sparse matrix
        void convert(const Matrix<dtype::real>& matrix, cudaStream_t stream);

    public:
        // matrix multiply
        void multiply(const Matrix<dtype::real>& matrix, cudaStream_t stream,
            Matrix<dtype::real>* result) const;

    public:
        // block size
        static const dtype::size block_size = 32;

    // accessors
    public:
        dtype::size rows() const { return this->rows_; }
        dtype::size columns() const { return this->columns_; }
        dtype::size data_rows() const { return this->data_rows_; }
        dtype::size data_columns() const { return this->data_columns_; }
        dtype::size density() const { return this->density_; }
        const dtype::real* values() const { return this->values_; }
        const dtype::index* column_ids() const { return this->column_ids_; }

    // mutators:
    public:
        dtype::real* values() { return this->values_; }
        dtype::index* column_ids() { return this->column_ids_; }
        dtype::size& density() { return this->density_; }

    // member
    private:
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
