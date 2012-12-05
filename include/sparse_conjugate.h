// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SPARSE_CONJUGATE_H
#define FASTEIT_INCLUDE_SPARSE_CONJUGATE_H

// namespace fastEIT
namespace fastEIT {
    // namespace numeric
    namespace numeric {
        // conjugate class definition
        class SparseConjugate {
        // constructor and destructor
        public:
            SparseConjugate(dtype::size rows, dtype::size columns,
                cudaStream_t stream);
            virtual ~SparseConjugate();

        public:
            // solve system
            void solve(const SparseMatrix& A, const Matrix<dtype::real>& f,
                dtype::size iterations, bool dcFree, cudaStream_t stream,
                Matrix<dtype::real>* x);

        public:
            // accessors
            dtype::size rows() const { return this->rows_; }
            dtype::size columns() const { return this->columns_; }
            const Matrix<dtype::real>& residuum() const { return *this->residuum_; }
            const Matrix<dtype::real>& projection() const { return *this->projection_; }
            const Matrix<dtype::real>& rsold() const { return *this->rsold_; }
            const Matrix<dtype::real>& rsnew() const { return *this->rsnew_; }
            const Matrix<dtype::real>& temp_vector() const { return *this->temp_vector_; }
            const Matrix<dtype::real>& temp_number() const { return *this->temp_number_; }

            // mutators
            Matrix<dtype::real>& residuum() { return *this->residuum_; }
            Matrix<dtype::real>& projection() { return *this->projection_; }
            Matrix<dtype::real>& rsold() { return *this->rsold_; }
            Matrix<dtype::real>& rsnew() { return *this->rsnew_; }
            Matrix<dtype::real>& temp_vector() { return *this->temp_vector_; }
            Matrix<dtype::real>& temp_number() { return *this->temp_number_; }

        // member
        private:
            dtype::size rows_;
            dtype::size columns_;
            Matrix<dtype::real>* residuum_;
            Matrix<dtype::real>* projection_;
            Matrix<dtype::real>* rsold_;
            Matrix<dtype::real>* rsnew_;
            Matrix<dtype::real>* temp_vector_;
            Matrix<dtype::real>* temp_number_;
        };
    }
}

#endif
