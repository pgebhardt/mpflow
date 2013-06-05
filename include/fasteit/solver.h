// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SOLVER_H
#define FASTEIT_INCLUDE_SOLVER_H

// namespace fastEIT
namespace fastEIT {
namespace solver {
    // class for solving differential EIT
    template <
        class numerical_forward_solver_type,
        class numerical_inverse_solver_type
    >
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<fastEIT::model::Model> model, dtype::index parallel_images,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<Matrix<dtype::real>> solve_differential(
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> solve_absolute(cublasHandle_t handle,
            cudaStream_t stream);

        // accessors
        std::shared_ptr<fastEIT::model::Model> model() { return this->model_; }
        std::shared_ptr<Forward<numerical_forward_solver_type>> forward_solver() {
            return this->forward_solver_;
        }
        std::shared_ptr<Inverse<numerical_inverse_solver_type>> inverse_solver() {
            return this->inverse_solver_;
        }
        std::shared_ptr<Matrix<dtype::real>> gamma() { return this->gamma_; }
        std::shared_ptr<Matrix<dtype::real>> dgamma() { return this->dgamma_; }
        std::shared_ptr<Matrix<dtype::real>> measured_voltage(dtype::index index) {
            return this->measured_voltage_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> calculated_voltage(dtype::index index) {
            return this->calculated_voltage_[index];
        }

    private:
        // member
        std::shared_ptr<fastEIT::model::Model> model_;
        std::shared_ptr<Forward<numerical_forward_solver_type>> forward_solver_;
        std::shared_ptr<Inverse<numerical_inverse_solver_type>> inverse_solver_;
        std::shared_ptr<Matrix<dtype::real>> gamma_;
        std::shared_ptr<Matrix<dtype::real>> dgamma_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> measured_voltage_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> calculated_voltage_;
    };
}
}

#endif
