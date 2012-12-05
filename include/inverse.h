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
    // constructor and destructor
    public:
        InverseSolver(dtype::size element_count, dtype::size voltage_count,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);
        virtual ~InverseSolver();

    public:
        // inverse solving
        const Matrix<dtype::real>& solve(const Matrix<dtype::real>& jacobian,
            const Matrix<dtype::real>& calculated_voltage,
            const Matrix<dtype::real>& measured_voltage,
            dtype::size steps, bool regularized, cublasHandle_t handle,
            cudaStream_t stream, Matrix<dtype::real>* gamma);

        // calc system matrix
        void calcSystemMatrix(const Matrix<dtype::real>& jacobian, cublasHandle_t handle,
            cudaStream_t stream);

        // calc excitation
        void calcExcitation(const Matrix<dtype::real>& jacobian,
            const Matrix<dtype::real>& calculated_voltage,
            const Matrix<dtype::real>& measured_voltage, cublasHandle_t handle,
            cudaStream_t stream);

    public:
        // accessors
        const NumericSolver& numeric_solver() const { return *this->numeric_solver_; }
        const Matrix<dtype::real>& dvoltage() const { return *this->dvoltage_; }
        const Matrix<dtype::real>& zeros() const { return *this->zeros_; }
        const Matrix<dtype::real>& excitation() const { return *this->excitation_; }
        const Matrix<dtype::real>& system_matrix() const { return *this->system_matrix_; }
        const Matrix<dtype::real>& jacobian_square() const { return *this->jacobian_square_; }
        const dtype::real& regularization_factor() const { return this->regularization_factor_; }

        // mutators
        NumericSolver& numeric_solver() { return *this->numeric_solver_; }
        Matrix<dtype::real>& dvoltage() { return *this->dvoltage_; }
        Matrix<dtype::real>& zeros() { return *this->zeros_; }
        Matrix<dtype::real>& excitation() { return *this->excitation_; }
        Matrix<dtype::real>& system_matrix() { return *this->system_matrix_; }
        Matrix<dtype::real>& jacobian_square() { return *this->jacobian_square_; }
        dtype::real& regularization_factor() { return this->regularization_factor_; }

    // member
    private:
        NumericSolver* numeric_solver_;
        Matrix<dtype::real>* dvoltage_;
        Matrix<dtype::real>* zeros_;
        Matrix<dtype::real>* excitation_;
        Matrix<dtype::real>* system_matrix_;
        Matrix<dtype::real>* jacobian_square_;
        dtype::real regularization_factor_;
    };
}

#endif
