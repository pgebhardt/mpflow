// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_EIT_INVERSE_H
#define MPFLOW_INCLDUE_EIT_INVERSE_H

// namespace mpFlow::EIT::solver
namespace mpFlow {
namespace EIT {
namespace solver {
    // inverse solver class definition
    template <
        class numerical_solver
    >
    class Inverse {
    public:
        // constructor
        Inverse(dtype::size element_count, dtype::size voltage_count, dtype::index parallel_images,
            dtype::real regularization_factor, cublasHandle_t handle, cudaStream_t stream);

    public:
        // inverse solving
        std::shared_ptr<Matrix<dtype::real>> solve(
            const std::shared_ptr<Matrix<dtype::real>> jacobian,
            const std::vector<std::shared_ptr<Matrix<dtype::real>>>& calculation,
            const std::vector<std::shared_ptr<Matrix<dtype::real>>>& measurement,
            dtype::size steps, cublasHandle_t handle, cudaStream_t stream,
            std::shared_ptr<Matrix<dtype::real>> gamma);

        // calc system matrix
        void calcSystemMatrix(const std::shared_ptr<Matrix<dtype::real>> jacobian,
            cublasHandle_t handle, cudaStream_t stream);

        // calc excitation
        void calcExcitation(const std::shared_ptr<Matrix<dtype::real>> jacobian,
            const std::vector<std::shared_ptr<Matrix<dtype::real>>>& calculation,
            const std::vector<std::shared_ptr<Matrix<dtype::real>>>& measurement,
            cublasHandle_t handle, cudaStream_t stream);

        // accessors
        std::shared_ptr<numerical_solver> numeric_solver() { return this->numeric_solver_; }
        std::shared_ptr<Matrix<dtype::real>> dvoltage() { return this->dvoltage_; }
        std::shared_ptr<Matrix<dtype::real>> zeros() { return this->zeros_; }
        std::shared_ptr<Matrix<dtype::real>> excitation() { return this->excitation_; }
        std::shared_ptr<Matrix<dtype::real>> system_matrix() { return this->system_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> jacobian_square() { return this->jacobian_square_; }
        dtype::real& regularization_factor() { return this->regularization_factor_; }

    private:
        // member
        std::shared_ptr<numerical_solver> numeric_solver_;
        std::shared_ptr<Matrix<dtype::real>> dvoltage_;
        std::shared_ptr<Matrix<dtype::real>> zeros_;
        std::shared_ptr<Matrix<dtype::real>> excitation_;
        std::shared_ptr<Matrix<dtype::real>> system_matrix_;
        std::shared_ptr<Matrix<dtype::real>> jacobian_square_;
        dtype::real regularization_factor_;
    };
}
}
}

#endif
