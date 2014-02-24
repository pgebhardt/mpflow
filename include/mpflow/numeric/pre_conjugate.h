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

#ifndef MPFLOW_INCLDUE_NUMERIC_PRE_CONJUGATE_H
#define MPFLOW_INCLDUE_NUMERIC_PRE_CONJUGATE_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // preconditioned conjugate gradient algorithm
    class PreConjugate {
    public:
        // constructor
        PreConjugate(dtype::size rows, dtype::size columns, cudaStream_t stream);

        // solve system
        void solve(const std::shared_ptr<Matrix<dtype::real>> A,
            const std::shared_ptr<Matrix<dtype::real>> f,
            dtype::size iterations, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> x);

        void set_preconditioner(std::shared_ptr<Matrix<dtype::real>> preconditioner) {
            this->preconditioner_ = preconditioner;
        }

        // accessors
        dtype::size rows() const { return this->rows_; }
        dtype::size columns() const { return this->columns_; }
        std::shared_ptr<Matrix<dtype::real>> residuum() { return this->residuum_; }
        std::shared_ptr<Matrix<dtype::real>> projection() { return this->projection_; }
        std::shared_ptr<Matrix<dtype::real>> z() { return this->z_; }
        std::shared_ptr<Matrix<dtype::real>> rsold() { return this->rsold_; }
        std::shared_ptr<Matrix<dtype::real>> rsnew() { return this->rsnew_; }
        std::shared_ptr<Matrix<dtype::real>> temp_vector() { return this->temp_vector_; }
        std::shared_ptr<Matrix<dtype::real>> temp_number() { return this->temp_number_; }
        std::shared_ptr<Matrix<dtype::real>> preconditioner() { return this->preconditioner_; }

    private:
        // member
        dtype::size rows_;
        dtype::size columns_;
        std::shared_ptr<Matrix<dtype::real>> residuum_;
        std::shared_ptr<Matrix<dtype::real>> projection_;
        std::shared_ptr<Matrix<dtype::real>> z_;
        std::shared_ptr<Matrix<dtype::real>> rsold_;
        std::shared_ptr<Matrix<dtype::real>> rsnew_;
        std::shared_ptr<Matrix<dtype::real>> temp_vector_;
        std::shared_ptr<Matrix<dtype::real>> temp_number_;
        std::shared_ptr<Matrix<dtype::real>> preconditioner_;
    };
}
}

#endif
