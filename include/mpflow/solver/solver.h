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

#ifndef MPFLOW_INCLDUE_SOLVER_SOLVER_H
#define MPFLOW_INCLDUE_SOLVER_SOLVER_H

// namespace mpFlow::solver
namespace mpFlow {
namespace solver {
    // class for solving differential EIT
    template <
        class forward_solver_type,
        template <template <class> class> class numerical_inverse_solver_type
    >
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<forward_solver_type> forward_solver, dtype::index parallel_images,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solve_differential(
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<numeric::Matrix<dtype::real>> solve_absolute(cublasHandle_t handle,
            cudaStream_t stream);

        // accessors
        std::shared_ptr<forward_solver_type> forward_solver() {
            return this->forward_solver_;
        }
        std::shared_ptr<Inverse<numerical_inverse_solver_type>> inverse_solver() {
            return this->inverse_solver_;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> gamma() { return this->gamma_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> dgamma() { return this->dgamma_; }
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& measurement() {
            return this->measurement_;
        }
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>>& calculation() {
            return this->calculation_;
        }

    private:
        // member
        std::shared_ptr<forward_solver_type> forward_solver_;
        std::shared_ptr<Inverse<numerical_inverse_solver_type>> inverse_solver_;
        std::shared_ptr<numeric::Matrix<dtype::real>> gamma_;
        std::shared_ptr<numeric::Matrix<dtype::real>> dgamma_;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> measurement_;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> calculation_;
    };
}
}

#endif
