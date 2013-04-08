// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FORWARD_H
#define FASTEIT_INCLUDE_FORWARD_H

// namespace fastEIT
namespace fastEIT {
    // forward solver class definition
    template <
        class numeric_solver_type
    >
    class ForwardSolver {
    public:
        // constructor
        ForwardSolver(std::shared_ptr<fastEIT::Model_base> model, cublasHandle_t handle,
            cudaStream_t stream);

        // apply pattern
        void applyMeasurementPattern(std::shared_ptr<Matrix<dtype::real>> result,
            cudaStream_t stream);

        // forward solving
        std::shared_ptr<Matrix<dtype::real>> solve(
            const std::shared_ptr<Matrix<dtype::real>> gamma, dtype::size steps,
            cublasHandle_t handle, cudaStream_t stream);

        // accessors
        std::shared_ptr<numeric_solver_type> numeric_solver() { return this->numeric_solver_; }
        std::shared_ptr<fastEIT::Model_base> model() { return this->model_; }
        std::shared_ptr<Matrix<dtype::real>> voltage() { return this->voltage_; }
        std::shared_ptr<Matrix<dtype::real>> current() { return this->current_; }

    private:
        // member
        std::shared_ptr<numeric_solver_type> numeric_solver_;
        std::shared_ptr<fastEIT::Model_base> model_;
        std::shared_ptr<Matrix<dtype::real>> voltage_;
        std::shared_ptr<Matrix<dtype::real>> current_;
    };
}

#endif
