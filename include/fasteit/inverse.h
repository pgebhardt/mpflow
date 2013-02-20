// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_INVERSE_H
#define FASTEIT_INCLUDE_INVERSE_H

// namespace fastEIT
namespace fastEIT {
    // inverse solver class definition
    template <
        class NumericSolver
    >
    class InverseSolver {
    public:
        // constructor
        InverseSolver(dtype::size element_count, dtype::size voltage_count,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

    public:
        // inverse solving
        std::shared_ptr<Matrix<dtype::real>> solve(
            const std::shared_ptr<Matrix<dtype::real>> jacobian,
            const std::shared_ptr<Matrix<dtype::real>> calculated_voltage,
            const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
            dtype::size steps, bool regularized, cublasHandle_t handle,
            cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> gamma);

        // calc system matrix
        void calcSystemMatrix(const std::shared_ptr<Matrix<dtype::real>> jacobian,
            cublasHandle_t handle, cudaStream_t stream);

        // calc excitation
        void calcExcitation(const std::shared_ptr<Matrix<dtype::real>> jacobian,
            const std::shared_ptr<Matrix<dtype::real>> calculated_voltage,
            const std::shared_ptr<Matrix<dtype::real>> measured_voltage,
            cublasHandle_t handle, cudaStream_t stream);

        // accessors
        const std::shared_ptr<NumericSolver> numeric_solver() const { return this->numeric_solver_; }
        const std::shared_ptr<Matrix<dtype::real>> dvoltage() const { return this->dvoltage_; }
        const std::shared_ptr<Matrix<dtype::real>> zeros() const { return this->zeros_; }
        const std::shared_ptr<Matrix<dtype::real>> excitation() const { return this->excitation_; }
        const std::shared_ptr<Matrix<dtype::real>> system_matrix() const { return this->system_matrix_; }
        const std::shared_ptr<Matrix<dtype::real>> jacobian_square() const { return this->jacobian_square_; }
        const dtype::real& regularization_factor() const { return this->regularization_factor_; }

        // mutators
        std::shared_ptr<NumericSolver> numeric_solver() { return this->numeric_solver_; }
        std::shared_ptr<Matrix<dtype::real>> dvoltage() { return this->dvoltage_; }
        std::shared_ptr<Matrix<dtype::real>> zeros() { return this->zeros_; }
        std::shared_ptr<Matrix<dtype::real>> excitation() { return this->excitation_; }
        std::shared_ptr<Matrix<dtype::real>> system_matrix() { return this->system_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> jacobian_square() { return this->jacobian_square_; }
        dtype::real& regularization_factor() { return this->regularization_factor_; }

    private:
        // member
        std::shared_ptr<NumericSolver> numeric_solver_;
        std::shared_ptr<Matrix<dtype::real>> dvoltage_;
        std::shared_ptr<Matrix<dtype::real>> zeros_;
        std::shared_ptr<Matrix<dtype::real>> excitation_;
        std::shared_ptr<Matrix<dtype::real>> system_matrix_;
        std::shared_ptr<Matrix<dtype::real>> jacobian_square_;
        dtype::real regularization_factor_;
    };
}

#endif
