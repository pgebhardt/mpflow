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
        class BasisFunction,
        class NumericSolver
    >
    class ForwardSolver {
    public:
        // constructor
        ForwardSolver(std::shared_ptr<Model<BasisFunction>> model,
            const std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
            const std::shared_ptr<Matrix<dtype::real>> drive_pattern,
            cublasHandle_t handle, cudaStream_t stream);

        // forward solving
        std::shared_ptr<Matrix<dtype::real>> solve(
            const std::shared_ptr<Matrix<dtype::real>> gamma, dtype::size steps,
            cublasHandle_t handle, cudaStream_t stream);

        // accessors
        const std::shared_ptr<Model<BasisFunction>> model() const { return this->model_; }
        const std::shared_ptr<NumericSolver> numeric_solver() const {
            return this->numeric_solver_;
        }
        const dtype::size& drive_count() const { return this->drive_count_; }
        const dtype::size& measurement_count() const  { return this->measurement_count_; }
        const std::shared_ptr<Matrix<dtype::real>> jacobian() const { return this->jacobian_; }
        const std::shared_ptr<Matrix<dtype::real>> voltage() const { return this->voltage_; }
        const std::shared_ptr<Matrix<dtype::real>> potential(dtype::index index) const {
            return this->potential_[index];
        }
        const std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) const {
            return this->excitation_[index];
        }
        const std::shared_ptr<Matrix<dtype::real>> voltage_calculation() const {
            return this->voltage_calculation_;
        }
        const std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix() const {
            return this->elemental_jacobian_matrix_;
        }

        // mutators
        std::shared_ptr<Model<BasisFunction>> model() { return this->model_; }
        std::shared_ptr<NumericSolver> numeric_solver() { return this->numeric_solver_; }
        dtype::size& drive_count() { return this->drive_count_; }
        dtype::size& measurement_count() { return this->measurement_count_; }
        std::shared_ptr<Matrix<dtype::real>> jacobian() { return this->jacobian_; }
        std::shared_ptr<Matrix<dtype::real>> voltage() { return this->voltage_; }
        std::shared_ptr<Matrix<dtype::real>> potential(dtype::index index) {
            return this->potential_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) {
            return this->excitation_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> voltage_calculation() { return this->voltage_calculation_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix() {
            return this->elemental_jacobian_matrix_;
        }

    private:
        // init jacobian calculation matrix
        void initJacobianCalculationMatrix(cublasHandle_t handle, cudaStream_t stream);

        // member
        std::shared_ptr<Model<BasisFunction>> model_;
        std::shared_ptr<NumericSolver> numeric_solver_;
        dtype::size drive_count_;
        dtype::size measurement_count_;
        std::shared_ptr<Matrix<dtype::real>> jacobian_;
        std::shared_ptr<Matrix<dtype::real>> voltage_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> potential_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> excitation_;
        std::shared_ptr<Matrix<dtype::real>> voltage_calculation_;
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix_;
    };
}

#endif
