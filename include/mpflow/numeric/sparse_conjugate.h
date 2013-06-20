// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_NUMERIC_SPARSE_CONJUGATE_H
#define MPFLOW_INCLDUE_NUMERIC_SPARSE_CONJUGATE_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // conjugate class definition
    class SparseConjugate {
    public:
        // constructor
        SparseConjugate(dtype::size rows, dtype::size columns,
            cudaStream_t stream);

        // solve system
        void solve(const std::shared_ptr<SparseMatrix<dtype::real>> A,
            const std::shared_ptr<Matrix<dtype::real>> f,
            dtype::size iterations, bool dcFree, cudaStream_t stream,
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
}
}

#endif
