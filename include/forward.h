// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FORWARD_H
#define FASTEIT_INCLUDE_FORWARD_H

// namespace fastEIT
namespace fastEIT {
    // forward declaration
    namespace forward {
        template <class, class> class SourcePolicy;
    }

    // forward solver class definition
    template <
        class numeric_solver_type,
        class model_type
    >
    class ForwardSolver
        : public forward::SourcePolicy<typename model_type::source_type,
            ForwardSolver<numeric_solver_type, model_type>> {
    public:
        // constructor
        ForwardSolver(std::shared_ptr<model_type> model,
            cublasHandle_t handle, cudaStream_t stream);

        // init excitation Matrix
        void initExcitationMatrix(cublasHandle_t handle, cudaStream_t stream);

        // accessors
        const std::shared_ptr<numeric_solver_type> numeric_solver() const {
            return this->numeric_solver_;
        }
        const std::shared_ptr<model_type> model() const { return this->model_; }
        const std::shared_ptr<Matrix<dtype::real>> jacobian() const { return this->jacobian_; }
        const std::shared_ptr<Matrix<dtype::real>> voltage() const { return this->voltage_; }
        const std::shared_ptr<Matrix<dtype::real>> current() const { return this->current_; }
        const std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) const {
            return this->excitation_[index];
        }
        const std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix() const {
            return this->elemental_jacobian_matrix_;
        }

        // mutators
        std::shared_ptr<numeric_solver_type> numeric_solver() { return this->numeric_solver_; }
        std::shared_ptr<model_type> model() { return this->model_; }
        std::shared_ptr<Matrix<dtype::real>> jacobian() { return this->jacobian_; }
        std::shared_ptr<Matrix<dtype::real>> voltage() { return this->voltage_; }
        std::shared_ptr<Matrix<dtype::real>> current() { return this->current_; }
        std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) {
            return this->excitation_[index];
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
        std::shared_ptr<Matrix<dtype::real>> jacobian_;
        std::shared_ptr<Matrix<dtype::real>> voltage_;
        std::shared_ptr<Matrix<dtype::real>> current_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> excitation_;
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix_;
    };

    namespace forward {
        // forward solver specialisation
        template <
            class source_type,
            class forward_solver_type
        >
        class SourcePolicy {
        };

        // specialisation for current source
        template <
            class forward_solver_type
        >
        class SourcePolicy<fastEIT::source::Current, forward_solver_type> {
        public:
            // constructor
            SourcePolicy(forward_solver_type* forward_solver)
                : forward_solver_(forward_solver) { }

            // forward solving
            std::shared_ptr<Matrix<dtype::real>> solve(
                const std::shared_ptr<Matrix<dtype::real>> gamma, dtype::size steps,
                cublasHandle_t handle, cudaStream_t stream);

        private:
            forward_solver_type* forward_solver_;
        };

        // specialisation for voltage source
        template <
            class forward_solver_type
        >
        class SourcePolicy<fastEIT::source::Voltage, forward_solver_type> {
        public:
            // constructor
            SourcePolicy(forward_solver_type* forward_solver)
                : forward_solver_(forward_solver) { }

            // forward solving
            std::shared_ptr<Matrix<dtype::real>> solve(
                const std::shared_ptr<Matrix<dtype::real>> gamma, dtype::size steps,
                cublasHandle_t handle, cudaStream_t stream);

        private:
            forward_solver_type* forward_solver_;
        };

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
