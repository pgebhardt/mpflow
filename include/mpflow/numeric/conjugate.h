// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_NUMERIC_CONJUGATE_H
#define MPFLOW_INCLDUE_NUMERIC_CONJUGATE_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // conjugate class definition
    class Conjugate {
    public:
        // constructor
        Conjugate(dtype::size rows, dtype::size columns, cudaStream_t stream);

        // solve system
        void solve(const std::shared_ptr<Matrix<dtype::real>> A,
            const std::shared_ptr<Matrix<dtype::real>> f,
            dtype::size iterations, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> x);

        // accessors
        dtype::size rows() const { return this->rows_; }
        dtype::size columns() const { return this->columns_; }
        const std::shared_ptr<Matrix<dtype::real>> residuum() const { return this->residuum_; }
        const std::shared_ptr<Matrix<dtype::real>> projection() const { return this->projection_; }
        const std::shared_ptr<Matrix<dtype::real>> rsold() const { return this->rsold_; }
        const std::shared_ptr<Matrix<dtype::real>> rsnew() const { return this->rsnew_; }
        const std::shared_ptr<Matrix<dtype::real>> temp_vector() const { return this->temp_vector_; }
        const std::shared_ptr<Matrix<dtype::real>> temp_number() const { return this->temp_number_; }

        // mutators
        std::shared_ptr<Matrix<dtype::real>> residuum() { return this->residuum_; }
        std::shared_ptr<Matrix<dtype::real>> projection() { return this->projection_; }
        std::shared_ptr<Matrix<dtype::real>> rsold() { return this->rsold_; }
        std::shared_ptr<Matrix<dtype::real>> rsnew() { return this->rsnew_; }
        std::shared_ptr<Matrix<dtype::real>> temp_vector() { return this->temp_vector_; }
        std::shared_ptr<Matrix<dtype::real>> temp_number() { return this->temp_number_; }

    private:
        // member
        dtype::size rows_;
        dtype::size columns_;
        std::shared_ptr<Matrix<dtype::real>> residuum_;
        std::shared_ptr<Matrix<dtype::real>> projection_;
        std::shared_ptr<Matrix<dtype::real>> rsold_;
        std::shared_ptr<Matrix<dtype::real>> rsnew_;
        std::shared_ptr<Matrix<dtype::real>> temp_vector_;
        std::shared_ptr<Matrix<dtype::real>> temp_number_;
    };

    // helper functions
    namespace conjugate {
        void addScalar(const std::shared_ptr<Matrix<dtype::real>> scalar,
            dtype::size rows, dtype::size columns, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> vector);

        void updateVector(const std::shared_ptr<Matrix<dtype::real>> x1,
            dtype::real sign, const std::shared_ptr<Matrix<dtype::real>> x2,
            const std::shared_ptr<Matrix<dtype::real>> r1,
            const std::shared_ptr<Matrix<dtype::real>> r2, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> result);

        void gemv(const std::shared_ptr<Matrix<dtype::real>> matrix,
            const std::shared_ptr<Matrix<dtype::real>> vector, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> result);
    }
}
}

#endif
