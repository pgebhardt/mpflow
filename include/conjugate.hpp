// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_CONJUGATE_HPP
#define FASTEIT_INCLUDE_CONJUGATE_HPP

// namespace fastEIT
namespace fastEIT {
    // namespace numeric
    namespace numeric {
        // conjugate class definition
        class Conjugate {
        // constructor and destructor
        public:
            Conjugate(dtype::size rows, cublasHandle_t handle, cudaStream_t stream);
            virtual ~Conjugate();

        public:
            // solve system
            void solve(const Matrix<dtype::real>& A, const Matrix<dtype::real>& f,
                dtype::size iterations, cublasHandle_t handle, cudaStream_t stream,
                Matrix<dtype::real>* x);

        public:
            // accessors
            const dtype::size& rows() const { return this->rows_; }
            const Matrix<dtype::real>& residuum() const { return *this->residuum_; }
            const Matrix<dtype::real>& projection() const { return *this->projection_; }
            const Matrix<dtype::real>& rsold() const { return *this->rsold_; }
            const Matrix<dtype::real>& rsnew() const { return *this->rsnew_; }
            const Matrix<dtype::real>& temp_vector() const { return *this->temp_vector_; }
            const Matrix<dtype::real>& temp_number() const { return *this->temp_number_; }

            // mutators
            dtype::size& rows() { return this->rows_; }
            Matrix<dtype::real>& residuum() { return *this->residuum_; }
            Matrix<dtype::real>& projection() { return *this->projection_; }
            Matrix<dtype::real>& rsold() { return *this->rsold_; }
            Matrix<dtype::real>& rsnew() { return *this->rsnew_; }
            Matrix<dtype::real>& temp_vector() { return *this->temp_vector_; }
            Matrix<dtype::real>& temp_number() { return *this->temp_number_; }

        // member
        private:
            dtype::size rows_;
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
