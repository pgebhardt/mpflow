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
        class BasisFunction
    >
    class Solver {
    // constructor and destructor
    public:
        Solver(Mesh<BasisFunction>* mesh, Electrodes* electrodes,
            const Matrix<dtype::real>& measurement_pattern,
            const Matrix<dtype::real>& drive_pattern, dtype::real sigma_ref,
            dtype::size num_harmonics, dtype::real regularization_factor,
            cublasHandle_t handle, cudaStream_t stream);
        virtual ~Solver();

    public:
        // pre solve for accurate initial jacobian
        void preSolve(cublasHandle_t handle, cudaStream_t stream);

        // calibrate
        Matrix<dtype::real>& calibrate(cublasHandle_t handle, cudaStream_t stream);
        Matrix<dtype::real>& calibrate(const Matrix<dtype::real>& calibration_voltage,
            cublasHandle_t handle, cudaStream_t stream);

        // solving
        Matrix<dtype::real>& solve(cublasHandle_t handle, cudaStream_t stream);
        Matrix<dtype::real>& solve(const Matrix<dtype::real>& measured_voltage,
            cublasHandle_t handle, cudaStream_t stream);
        Matrix<dtype::real>& solve(const Matrix<dtype::real>& measured_voltage,
            const Matrix<dtype::real>& calibration_voltage, cublasHandle_t handle,
            cudaStream_t stream);

    public:
        // accessors
        const ForwardSolver<BasisFunction, numeric::SparseConjugate>& forward_solver() const {
            return *this->forward_solver_;
        }
        const InverseSolver<numeric::Conjugate>& inverse_solver() const {
            return *this->inverse_solver_;
        }
        const Matrix<dtype::real>& dgamma() const { return *this->dgamma_; }
        const Matrix<dtype::real>& gamma() const { return *this->gamma_; }
        const Matrix<dtype::real>& measured_voltage() const { return *this->measured_voltage_; }
        const Matrix<dtype::real>& calibration_voltage() const {
            return *this->calibration_voltage_;
        }

        // mutators
        ForwardSolver<BasisFunction, numeric::SparseConjugate>& forward_solver() {
            return *this->forward_solver_;
        }
        InverseSolver<numeric::Conjugate>& inverse_solver() { return *this->inverse_solver_; }
        Matrix<dtype::real>& dgamma() { return *this->dgamma_; }
        Matrix<dtype::real>& gamma() { return *this->gamma_; }
        Matrix<dtype::real>& measured_voltage() { return *this->measured_voltage_; }
        Matrix<dtype::real>& calibration_voltage() { return *this->calibration_voltage_; }

    // member
    private:
        Model<BasisFunction>* model_;
        ForwardSolver<BasisFunction, numeric::SparseConjugate>* forward_solver_;
        InverseSolver<numeric::Conjugate>* inverse_solver_;
        Matrix<dtype::real>* dgamma_;
        Matrix<dtype::real>* gamma_;
        Matrix<dtype::real>* measured_voltage_;
        Matrix<dtype::real>* calibration_voltage_;
    };
}

#endif
