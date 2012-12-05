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
    // constructor and destructor
    public:
        ForwardSolver(Mesh<BasisFunction>* mesh, Electrodes* electrodes,
            const Matrix<dtype::real>& measurment_pattern,
            const Matrix<dtype::real>& drive_pattern, dtype::real sigma_ref,
            dtype::size num_harmonics, cublasHandle_t handle, cudaStream_t stream);
        virtual ~ForwardSolver();

    public:
        // forward solving
        Matrix<dtype::real>& solve(const Matrix<dtype::real>& gamma, dtype::size steps,
            cublasHandle_t handle, cudaStream_t stream);

    protected:
        // init jacobian calculation matrix
        void initJacobianCalculationMatrix(cublasHandle_t handle, cudaStream_t stream);

    public:
        // accessors
        const Model<BasisFunction>& model() const { return *this->model_; }
        const NumericSolver& numeric_solver() const { return *this->numeric_solver_; }
        dtype::size drive_count() const { return this->drive_count_; }
        dtype::size measurment_count() const { return this->measurment_count_; }
        const Matrix<dtype::real>& jacobian() const { return *this->jacobian_; }
        const Matrix<dtype::real>& voltage() const { return *this->voltage_; }
        const Matrix<dtype::real>& potential(dtype::index index) const { return *this->potential_[index]; }
        const Matrix<dtype::real>& excitation(dtype::index index) const {
            return *this->excitation_[index]; }
        const Matrix<dtype::real>& voltage_calculation() const {
            return *this->voltage_calculation_; }
        const Matrix<dtype::real>& elemental_jacobian_matrix() const {
            return *this->elemental_jacobian_matrix_; }

        // mutators
        Model<BasisFunction>& model() { return *this->model_; }
        NumericSolver& numeric_solver() { return *this->numeric_solver_; }
        Matrix<dtype::real>& jacobian() { return *this->jacobian_; }
        Matrix<dtype::real>& voltage() { return *this->voltage_; }
        Matrix<dtype::real>& potential(dtype::index index) { return *this->potential_[index]; }
        Matrix<dtype::real>& excitation(dtype::index index) {
            return *this->excitation_[index]; }
        Matrix<dtype::real>& voltage_calculation() { return *this->voltage_calculation_; }
        Matrix<dtype::real>& elemental_jacobian_matrix() {
            return *this->elemental_jacobian_matrix_; }

    // member
    private:
        Model<BasisFunction>* model_;
        NumericSolver* numeric_solver_;
        dtype::size drive_count_;
        dtype::size measurment_count_;
        Matrix<dtype::real>* jacobian_;
        Matrix<dtype::real>* voltage_;
        std::vector<Matrix<dtype::real>*> potential_;
        std::vector<Matrix<dtype::real>*> excitation_;
        Matrix<dtype::real>* voltage_calculation_;
        Matrix<dtype::real>* elemental_jacobian_matrix_;
    };
}

#endif
