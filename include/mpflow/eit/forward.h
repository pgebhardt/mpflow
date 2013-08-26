// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_EIT_FORWARD_H
#define MPFLOW_INCLDUE_EIT_FORWARD_H

// namespace mpFlow::EIT::solver
namespace mpFlow {
namespace EIT {
namespace solver {
    // forward solver class definition
    template <
        class numerical_solver
    >
    class Forward {
    public:
        // constructor
        Forward(std::shared_ptr<mpFlow::EIT::model::Base> model, cudaStream_t stream);

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
}

#endif
