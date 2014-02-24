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

#ifndef MPFLOW_INCLDUE_SOLVER_INVERSE_H
#define MPFLOW_INCLDUE_SOLVER_INVERSE_H

// namespace mpFlow::solver
namespace mpFlow {
namespace solver {
    // inverse solver class definition
    template <
        class numerical_solver
    >
    class Inverse {
    public:
        // constructor
        Inverse(dtype::size element_count, dtype::size voltage_count, dtype::index parallel_images,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

    public:
        // inverse solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solve(
            const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement,
            dtype::size steps, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<numeric::Matrix<dtype::real>> gamma);

        // calc system matrix
        void calcSystemMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
            cublasHandle_t handle, cudaStream_t stream);

        // calc excitation
        void calcExcitation(const std::shared_ptr<numeric::Matrix<dtype::real>> jacobian,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation,
            const std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement,
            cublasHandle_t handle, cudaStream_t stream);

        // accessors
        std::shared_ptr<numerical_solver> numeric_solver() { return this->numeric_solver_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> difference() { return this->difference_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> zeros() { return this->zeros_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> excitation() { return this->excitation_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> system_matrix() { return this->system_matrix_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> jacobian_square() { return this->jacobian_square_; }
        dtype::real& regularization_factor() { return this->regularization_factor_; }

    private:
        // member
        std::shared_ptr<numerical_solver> numeric_solver_;
        std::shared_ptr<numeric::Matrix<dtype::real>> difference_;
        std::shared_ptr<numeric::Matrix<dtype::real>> zeros_;
        std::shared_ptr<numeric::Matrix<dtype::real>> excitation_;
        std::shared_ptr<numeric::Matrix<dtype::real>> system_matrix_;
        std::shared_ptr<numeric::Matrix<dtype::real>> jacobian_square_;
        dtype::real regularization_factor_;
    };
}
}

#endif
