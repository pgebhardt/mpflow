// fastEIT
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MODEL_H
#define FASTEIT_INCLUDE_MODEL_H

// namespaces fastEIT
namespace fastEIT {
    // model base class
    class Model_base {
    public:
        // constructor
        Model_base(std::shared_ptr<Mesh> mesh, std::shared_ptr<Electrodes> electrodes,
            std::shared_ptr<source::Source> source,
            dtype::real sigmaRef, dtype::size components_count);

        // update model
        virtual void update(const std::shared_ptr<Matrix<dtype::real>>,
            cublasHandle_t, cudaStream_t) {
        }

        // calc jacobian
        virtual std::shared_ptr<Matrix<dtype::real>> calcJacobian(
            const std::shared_ptr<Matrix<dtype::real>>, cudaStream_t) {
            return nullptr;
        }

        // accessors
        std::shared_ptr<Mesh> mesh() { return this->mesh_; }
        std::shared_ptr<Electrodes> electrodes() { return this->electrodes_; }
        std::shared_ptr<source::Source> source() {
            return this->source_;
        }
        std::shared_ptr<SparseMatrix<dtype::real>> system_matrix(dtype::index index) {
            return this->system_matrices_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> potential(dtype::index index) {
            return this->potential_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> jacobian() { return this->jacobian_; }
        std::shared_ptr<SparseMatrix<dtype::real>> s_matrix() { return this->s_matrix_; }
        std::shared_ptr<SparseMatrix<dtype::real>> r_matrix() { return this->r_matrix_; }
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix() {
            return this->connectivity_matrix_;
        }
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix() {
            return this->elemental_s_matrix_;
        }
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix() {
            return this->elemental_r_matrix_;
        }
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix() {
            return this->elemental_jacobian_matrix_;
        }
        dtype::real sigma_ref() { return this->sigma_ref_; }
        dtype::size components_count() { return this->components_count_; }

    protected:
        // init methods
        virtual void init(cublasHandle_t, cudaStream_t) { }
        virtual std::shared_ptr<Matrix<dtype::real>> initElementalMatrices(cudaStream_t) {
            return nullptr;
        }
        virtual void initJacobianCalculationMatrix(cublasHandle_t, cudaStream_t) {
        }

        // member
        std::shared_ptr<Mesh> mesh_;
        std::shared_ptr<Electrodes> electrodes_;
        std::shared_ptr<source::Source> source_;
        std::vector<std::shared_ptr<SparseMatrix<dtype::real>>> system_matrices_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> potential_;
        std::shared_ptr<Matrix<dtype::real>> jacobian_;
        std::shared_ptr<SparseMatrix<dtype::real>> s_matrix_;
        std::shared_ptr<SparseMatrix<dtype::real>> r_matrix_;
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix_;
        dtype::real sigma_ref_;
        dtype::size components_count_;
    };

    // model class definition
    template <
        class basis_function_type
    >
    class Model :
    public Model_base {
    public:
        // constructor
        Model(std::shared_ptr<Mesh> mesh, std::shared_ptr<Electrodes> electrodes,
            std::shared_ptr<source::Source> source, dtype::real sigmaRef,
            dtype::size components_count, cublasHandle_t handle, cudaStream_t stream);

        // update model
        virtual void update(const std::shared_ptr<Matrix<dtype::real>> gamma,
            cublasHandle_t handle, cudaStream_t stream);

        // calc jacobian
        virtual std::shared_ptr<Matrix<dtype::real>> calcJacobian(
            const std::shared_ptr<Matrix<dtype::real>> gamma, cudaStream_t stream);

    protected:
        // init methods
        virtual void init(cublasHandle_t handle, cudaStream_t stream);
        virtual std::shared_ptr<Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);
        virtual void initJacobianCalculationMatrix(cublasHandle_t handle, cudaStream_t stream);
    };

    // special functions
    namespace model {
        // update matrix
        void updateMatrix(const std::shared_ptr<Matrix<dtype::real>> elements,
            const std::shared_ptr<Matrix<dtype::real>> gamma,
            const std::shared_ptr<Matrix<dtype::index>> connectivityMatrix, dtype::real sigmaRef,
            cudaStream_t stream, std::shared_ptr<SparseMatrix<dtype::real>> matrix);

        // reduce matrix
        template <
            class type
        >
        void reduceMatrix(const std::shared_ptr<Matrix<type>> intermediateMatrix,
            const std::shared_ptr<SparseMatrix<dtype::real>> shape, dtype::index offset,
            cudaStream_t stream, std::shared_ptr<Matrix<type>> matrix);

        // calc jacobian
        template <
            class basis_function_type
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
