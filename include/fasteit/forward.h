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
        class numeric_solver_type,
        class model_type
    >
    class ForwardSolver {
    public:
        // constructor
        ForwardSolver(std::shared_ptr<model_type> model,
            std::shared_ptr<source::Source<model_type>> source, cublasHandle_t handle,
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
        std::shared_ptr<model_type> model() { return this->model_; }
        std::shared_ptr<source::Source<model_type>> source() { return this->source_; }
        std::shared_ptr<Matrix<dtype::real>> jacobian() { return this->jacobian_; }
        std::shared_ptr<Matrix<dtype::real>> voltage() { return this->voltage_; }
        std::shared_ptr<Matrix<dtype::real>> current() { return this->current_; }
        std::shared_ptr<Matrix<dtype::real>> potential(dtype::index index) {
            return this->potential_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix() {
            return this->elemental_jacobian_matrix_;
        }

    private:
        // init jacobian calculation matrix
        void initJacobianCalculationMatrix(cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<numeric_solver_type> numeric_solver_;
        std::shared_ptr<model_type> model_;
        std::shared_ptr<source::Source<model_type>> source_;
        std::shared_ptr<Matrix<dtype::real>> jacobian_;
        std::shared_ptr<Matrix<dtype::real>> voltage_;
        std::shared_ptr<Matrix<dtype::real>> current_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> potential_;
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix_;
    };

    namespace forward {
        // calc jacobian
        template <
            class model_type
        >
        void calcJacobian(const std::shared_ptr<Matrix<dtype::real>> gamma,
            const std::shared_ptr<Matrix<dtype::real>> potential,
            const std::shared_ptr<Matrix<dtype::index>> elements,
            const std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix,
            dtype::size drive_count, dtype::size measurment_count,
            dtype::real sigma_ref, bool additiv,
            cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> jacobian);
    }
}

#endif
