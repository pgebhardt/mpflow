// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MODEL_H
#define FASTEIT_INCLUDE_MODEL_H

// namespaces fastEIT
namespace fastEIT {
    // model class definition
    template <
        class BasisFunction
    >
    class Model {
    public:
        // constructor
        Model(std::shared_ptr<Mesh<BasisFunction>> mesh, std::shared_ptr<Electrodes<Mesh<BasisFunction>>> electrodes,
            dtype::real sigmaRef, dtype::size components_count, cublasHandle_t handle, cudaStream_t stream);

        // calc nodal current density
        void calcNodalCurrentDensity(const std::shared_ptr<Matrix<dtype::real>> current_density, dtype::size component,
            cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> nodal_current_density);

        // update model
        void update(const std::shared_ptr<Matrix<dtype::real>> gamma, cublasHandle_t handle, cudaStream_t stream);

        // basis function type
        typedef BasisFunction basis_function_type;

        // accessors
        const std::shared_ptr<Mesh<BasisFunction>> mesh() const { return this->mesh_; }
        const std::shared_ptr<Electrodes<Mesh<BasisFunction>>> electrodes() const { return this->electrodes_; }
        dtype::real sigma_ref() const { return this->sigma_ref_; }
        const std::shared_ptr<SparseMatrix> system_matrix(dtype::index index) const { return this->system_matrices_[index]; }
        const std::shared_ptr<Matrix<dtype::real>> excitation_matrix() const { return this->excitation_matrix_; }
        dtype::size components_count() const { return this->components_count_; }
        const std::shared_ptr<SparseMatrix> s_matrix() const { return this->s_matrix_; }
        const std::shared_ptr<SparseMatrix> r_matrix() const { return this->r_matrix_; }
        const std::shared_ptr<Matrix<dtype::index>> connectivity_matrix() const { return this->connectivity_matrix_; }
        const std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix() const { return this->elemental_s_matrix_; }
        const std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix() const { return this->elemental_r_matrix_; }

        // mutators
        std::shared_ptr<Mesh<BasisFunction>> mesh() { return this->mesh_; }
        std::shared_ptr<Electrodes<Mesh<BasisFunction>>> electrodes() { return this->electrodes_; }
        dtype::real sigma_ref() { return this->sigma_ref_; }
        std::shared_ptr<SparseMatrix> system_matrix(dtype::index index) { return this->system_matrices_[index]; }
        std::shared_ptr<Matrix<dtype::real>> excitation_matrix() { return this->excitation_matrix_; }
        dtype::size components_count() { return this->components_count_; }
        std::shared_ptr<SparseMatrix> s_matrix() { return this->s_matrix_; }
        std::shared_ptr<SparseMatrix> r_matrix() { return this->r_matrix_; }
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix() { return this->connectivity_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix() { return this->elemental_s_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix() { return this->elemental_r_matrix_; }

    private:
        // init methods
        void init(cublasHandle_t handle, cudaStream_t stream);
        void createSparseMatrices(cublasHandle_t handle, cudaStream_t stream);
        void initExcitationMatrix(cudaStream_t stream);

        // member
        std::shared_ptr<Mesh<BasisFunction>> mesh_;
        std::shared_ptr<Electrodes<Mesh<BasisFunction>>> electrodes_;
        dtype::real sigma_ref_;
        std::vector<std::shared_ptr<SparseMatrix>> system_matrices_;
        std::shared_ptr<SparseMatrix> s_matrix_;
        std::shared_ptr<SparseMatrix> r_matrix_;
        std::shared_ptr<Matrix<dtype::real>> excitation_matrix_;
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix_;
        dtype::size components_count_;
    };

    // special functions
    namespace model {
        // update matrix
        void updateMatrix(const std::shared_ptr<Matrix<dtype::real>> elements,
            const std::shared_ptr<Matrix<dtype::real>> gamma,
            const std::shared_ptr<Matrix<dtype::index>> connectivityMatrix, dtype::real sigmaRef,
            cudaStream_t stream, std::shared_ptr<SparseMatrix> matrix);

        // reduce matrix
        template <
            class type
        >
        void reduceMatrix(const std::shared_ptr<Matrix<type>> intermediateMatrix,
            const std::shared_ptr<SparseMatrix> shape, cudaStream_t stream,
            std::shared_ptr<Matrix<type>> matrix);
    }
}

#endif
