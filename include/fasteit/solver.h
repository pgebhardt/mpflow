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
    class Differential {
    public:
        // constructor
        Differential(std::shared_ptr<fastEIT::model::Model> model, dtype::index parallel_images,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<Matrix<dtype::real>> solve(cublasHandle_t handle, cudaStream_t stream);

        // accessors
        std::shared_ptr<fastEIT::model::Model> model() { return this->model_; }
        std::shared_ptr<Forward<numeric::SparseConjugate>> forward_solver() {
            return this->forward_solver_;
        }
        std::shared_ptr<Inverse<numeric::Conjugate>> inverse_solver() {
            return this->inverse_solver_;
        }
        std::shared_ptr<Matrix<dtype::real>> dgamma() { return this->dgamma_; }
        std::shared_ptr<Matrix<dtype::real>> measured_voltage(dtype::index index) {
            return this->measured_voltage_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> calibration_voltage(dtype::index index) {
            return this->calibration_voltage_[index];
        }

    private:
        // member
        std::shared_ptr<fastEIT::model::Model> model_;
        std::shared_ptr<Forward<numeric::SparseConjugate>> forward_solver_;
        std::shared_ptr<Inverse<numeric::Conjugate>> inverse_solver_;
        std::shared_ptr<Matrix<dtype::real>> dgamma_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> measured_voltage_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> calibration_voltage_;
    };

    // class for solving absolute EIT
    class Absolute {
    public:
        // constructor
        Absolute(std::shared_ptr<fastEIT::model::Model> model, dtype::real regularization_factor,
            cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<Matrix<dtype::real>> solve(const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> solve(cublasHandle_t handle, cudaStream_t stream);

        // accessors
        std::shared_ptr<fastEIT::model::Model> model() { return this->model_; }
        std::shared_ptr<Forward<numeric::SparseConjugate>> forward_solver() {
            return this->forward_solver_;
        }
        std::shared_ptr<Inverse<numeric::FastConjugate>> inverse_solver() {
            return this->inverse_solver_;
        }
        std::shared_ptr<Matrix<dtype::real>> dgamma() { return this->dgamma_; }
        std::shared_ptr<Matrix<dtype::real>> gamma() { return this->gamma_; }
        std::shared_ptr<Matrix<dtype::real>> measured_voltage() {
            return this->measured_voltage_[0];
        }
        std::shared_ptr<Matrix<dtype::real>> calculated_voltage() {
            return this->calculated_voltage_[0];
        }

    private:
        // member
        std::shared_ptr<fastEIT::model::Model> model_;
        std::shared_ptr<Forward<numeric::SparseConjugate>> forward_solver_;
        std::shared_ptr<Inverse<numeric::FastConjugate>> inverse_solver_;
        std::shared_ptr<Matrix<dtype::real>> dgamma_;
        std::shared_ptr<Matrix<dtype::real>> gamma_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> measured_voltage_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> calculated_voltage_;
    };
}
}

#endif
