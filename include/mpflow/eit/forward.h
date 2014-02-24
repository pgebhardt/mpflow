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

#ifndef MPFLOW_INCLDUE_EIT_FORWARDSOLVER_H
#define MPFLOW_INCLDUE_EIT_FORWARDSOLVER_H

// namespace mpFlow::EIT::solver
namespace mpFlow {
namespace EIT {
    // forward solver class definition
    template <
        class numerical_solver
    >
    class ForwardSolver {
    public:
        // constructor
        ForwardSolver(std::shared_ptr<mpFlow::EIT::model::Base> model, cudaStream_t stream);

        // apply pattern
        void applyMeasurementPattern(std::shared_ptr<numeric::Matrix<dtype::real>> result,
            cudaStream_t stream);

        // forward solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solve(
            const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, dtype::size steps,
            cudaStream_t stream);

        // accessors
        std::shared_ptr<numerical_solver> numeric_solver() { return this->numeric_solver_; }
        std::shared_ptr<mpFlow::EIT::model::Base> model() { return this->model_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> voltage() { return this->voltage_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> current() { return this->current_; }

    private:
        // member
        std::shared_ptr<numerical_solver> numeric_solver_;
        std::shared_ptr<mpFlow::EIT::model::Base> model_;
        std::shared_ptr<numeric::Matrix<dtype::real>> voltage_;
        std::shared_ptr<numeric::Matrix<dtype::real>> current_;
    };
}
}

#endif
