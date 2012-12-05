// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MATRIX_H
#define FASTEIT_INCLUDE_MATRIX_H

// namespace fastEIT
namespace fastEIT {
    // matrix class definition
    template <
        class type
    >
    class Matrix {
    public:
        // constructor and destructor
        Matrix(dtype::size rows, dtype::size columns, cudaStream_t stream);
        virtual ~Matrix();

        // block size
        static const dtype::size block_size = 16;

        // copy methods
        void copy(const Matrix<type>& other, cudaStream_t stream);
        void copyToDevice(cudaStream_t stream);
        void copyToHost(cudaStream_t stream);

        // mathematical methods
        void add(const Matrix<type>& value, cudaStream_t stream);
        void multiply(const Matrix<type>& A, const Matrix<type>& B, cublasHandle_t handle, cudaStream_t stream);
        void scalarMultiply(type scalar, cudaStream_t stream);
        void vectorDotProduct(const Matrix<type>& A, const Matrix<type>& B, cudaStream_t stream);

        // reduce methods
        void sum(const Matrix<type>& value, cudaStream_t stream);
        void min(const Matrix<type>& value, cudaStream_t stream);
        void max(const Matrix<type>& value, cudaStream_t stream);

        // accessors
        const type* host_data() const { return this->host_data_; }
        const type* device_data() const { return this->device_data_; }
        dtype::size rows() const { return this->rows_; }
        dtype::size columns() const { return this->columns_; }
        dtype::size data_rows() const { return this->data_rows_; }
        dtype::size data_columns() const { return this->data_columns_; }
        const type& operator() (dtype::index i, dtype::index j) const {
            assert(i < this->rows());
            assert(j < this->columns());
            return this->host_data_[i + j * this->data_rows()];
        }

        // mutators
        type* device_data() { return this->device_data_; }
        type& operator() (dtype::index i, dtype::index j) {
            assert(i < this->rows());
            assert(j < this->columns());
            return this->host_data_[i + j * this->data_rows()];
        }
        type* host_data() { return this->host_data_; }

    // member
    private:
        type* host_data_;
        type* device_data_;
        dtype::size rows_;
        dtype::size columns_;
        dtype::size data_rows_;
        dtype::size data_columns_;
    };

    // namespace matrix
    namespace matrix {
        // load matrix from file
        template <
            class type
        >
        fastEIT::Matrix<type>* loadtxt(const std::string filename, cudaStream_t stream);

        // save matrix to file
        template <
            class type
        >
        void savetxt(const std::string filename, const Matrix<type>& matrix);
    }
}

#endif
