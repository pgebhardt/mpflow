// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_EIT_SOLVER_H
#define MPFLOW_INCLDUE_EIT_SOLVER_H

// namespace mpFlow::EIT::solver
namespace mpFlow {
namespace EIT {
namespace solver {
    // class for solving differential EIT
    template <
        class numerical_forward_solver_type,
        class numerical_inverse_solver_type
    >
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<mpFlow::EIT::model::Base> model, dtype::index parallel_images,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<numeric::Matrix<dtype::real>> solve_differential(
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<numeric::Matrix<dtype::real>> solve_absolute(cublasHandle_t handle,
            cudaStream_t stream);

        // accessors
        std::shared_ptr<mpFlow::EIT::model::Base> model() { return this->model_; }
        std::shared_ptr<Forward<numerical_forward_solver_type>> forward_solver() {
            return this->forward_solver_;
        }
        std::shared_ptr<Inverse<numerical_inverse_solver_type>> inverse_solver() {
            return this->inverse_solver_;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> gamma() { return this->gamma_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> dgamma() { return this->dgamma_; }
        std::shared_ptr<numeric::Matrix<dtype::real>> measurement(dtype::index index) {
            return this->measurement_[index];
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> calculation(dtype::index index) {
            return this->calculation_[index];
        }

    private:
        // member
        std::shared_ptr<mpFlow::EIT::model::Base> model_;
        std::shared_ptr<Forward<numerical_forward_solver_type>> forward_solver_;
        std::shared_ptr<Inverse<numerical_inverse_solver_type>> inverse_solver_;
        std::shared_ptr<numeric::Matrix<dtype::real>> gamma_;
        std::shared_ptr<numeric::Matrix<dtype::real>> dgamma_;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> measurement_;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> calculation_;
    };
}
}
}

#endif
