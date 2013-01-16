// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SOLVER_H
#define FASTEIT_INCLUDE_SOLVER_H

// namespace fastEIT
namespace fastEIT {
    // solver class definition
    template <
        class model_type
    >
    class Solver {
    public:
        // constructor
        Solver(std::shared_ptr<model_type> model, dtype::real regularization_factor,
            cublasHandle_t handle, cudaStream_t stream);

        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // calibrate
        std::shared_ptr<Matrix<dtype::real>> calibrate(cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> calibrate(const std::shared_ptr<Matrix<dtype::real>> calibration_voltage,
            cublasHandle_t handle, cudaStream_t stream);

        // solving
        std::shared_ptr<Matrix<dtype::real>> solve(cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> solve(const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
            cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> solve(const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
            const std::shared_ptr<Matrix<dtype::real>> calibration_voltage, cublasHandle_t handle,
            cudaStream_t stream);

        // accessors
        const std::shared_ptr<model_type> model() const { return this->model_; }
        const std::shared_ptr<ForwardSolver<numeric::SparseConjugate, model_type>> forward_solver() const {
            return this->forward_solver_;
        }
        const std::shared_ptr<InverseSolver<numeric::Conjugate>> inverse_solver() const {
            return this->inverse_solver_;
        }
        const std::shared_ptr<Matrix<dtype::real>> dgamma() const { return this->dgamma_; }
        const std::shared_ptr<Matrix<dtype::real>> gamma() const { return this->gamma_; }
        const std::shared_ptr<Matrix<dtype::real>> measured_voltage() const { return this->measured_voltage_; }
        const std::shared_ptr<Matrix<dtype::real>> calibration_voltage() const { return this->calibration_voltage_; }

        // mutators
        std::shared_ptr<model_type> model() { return this->model_; }
        std::shared_ptr<ForwardSolver<numeric::SparseConjugate, model_type>> forward_solver() {
            return this->forward_solver_;
        }
        std::shared_ptr<InverseSolver<numeric::Conjugate>> inverse_solver() {
            return this->inverse_solver_;
        }
        std::shared_ptr<Matrix<dtype::real>> dgamma() { return this->dgamma_; }
        std::shared_ptr<Matrix<dtype::real>> gamma() { return this->gamma_; }
        std::shared_ptr<Matrix<dtype::real>> measured_voltage() { return this->measured_voltage_; }
        std::shared_ptr<Matrix<dtype::real>> calibration_voltage() { return this->calibration_voltage_; }

    private:
        // member
        std::shared_ptr<model_type> model_;
        std::shared_ptr<ForwardSolver<numeric::SparseConjugate, model_type>> forward_solver_;
        std::shared_ptr<InverseSolver<numeric::Conjugate>> inverse_solver_;
        std::shared_ptr<Matrix<dtype::real>> dgamma_;
        std::shared_ptr<Matrix<dtype::real>> gamma_;
        std::shared_ptr<Matrix<dtype::real>> measured_voltage_;
        std::shared_ptr<Matrix<dtype::real>> calibration_voltage_;
    };
}

#endif
