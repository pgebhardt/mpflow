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
        class template_basis_function_type
    >
    class Model {
    public:
        // constructor
        Model(
            std::shared_ptr<Mesh<template_basis_function_type>> mesh,
            std::shared_ptr<Electrodes<Mesh<template_basis_function_type>>> electrodes,
            std::shared_ptr<source::Source> source, dtype::real sigmaRef,
            dtype::size components_count, cublasHandle_t handle, cudaStream_t stream);

        // update model
        void update(const std::shared_ptr<Matrix<dtype::real>> gamma, cublasHandle_t handle, cudaStream_t stream);

        // type defs
        typedef template_basis_function_type basis_function_type;

        // accessors
        const std::shared_ptr<Mesh<template_basis_function_type>> mesh() const { return this->mesh_; }
        const std::shared_ptr<Electrodes<Mesh<template_basis_function_type>>> electrodes() const { return this->electrodes_; }
        const std::shared_ptr<source::Source> source() const { return this->source_; }
        const std::shared_ptr<Matrix<dtype::real>> potential(dtype::index index) const {
            return this->potential_[index];
        }
        const std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) const {
            return this->excitation_[index];
        }
        const std::shared_ptr<SparseMatrix> system_matrix(dtype::index index) const { return this->system_matrices_[index]; }
        const std::shared_ptr<SparseMatrix> s_matrix() const { return this->s_matrix_; }
        const std::shared_ptr<SparseMatrix> r_matrix() const { return this->r_matrix_; }
        const std::shared_ptr<Matrix<dtype::index>> connectivity_matrix() const { return this->connectivity_matrix_; }
        const std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix() const { return this->elemental_s_matrix_; }
        const std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix() const { return this->elemental_r_matrix_; }
        const std::shared_ptr<Matrix<dtype::real>> excitation_matrix() const { return this->excitation_matrix_; }
        dtype::real sigma_ref() const { return this->sigma_ref_; }
        dtype::size components_count() const { return this->components_count_; }

        // mutators
        std::shared_ptr<Mesh<template_basis_function_type>> mesh() { return this->mesh_; }
        std::shared_ptr<Electrodes<Mesh<template_basis_function_type>>> electrodes() { return this->electrodes_; }
        std::shared_ptr<source::Source> source() { return this->source_; }
        std::shared_ptr<Matrix<dtype::real>> potential(dtype::index index) {
            return this->potential_[index];
        }
        std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) {
            return this->excitation_[index];
        }
        std::shared_ptr<SparseMatrix> system_matrix(dtype::index index) { return this->system_matrices_[index]; }
        std::shared_ptr<SparseMatrix> s_matrix() { return this->s_matrix_; }
        std::shared_ptr<SparseMatrix> r_matrix() { return this->r_matrix_; }
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix() { return this->connectivity_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix() { return this->elemental_s_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix() { return this->elemental_r_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> excitation_matrix() { return this->excitation_matrix_; }
        dtype::real sigma_ref() { return this->sigma_ref_; }
        dtype::size components_count() { return this->components_count_; }

    private:
        // init methods
        void init(cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);
        void initExcitation(cublasHandle_t handle, cudaStream_t);

        // member
        std::shared_ptr<Mesh<template_basis_function_type>> mesh_;
        std::shared_ptr<Electrodes<Mesh<template_basis_function_type>>> electrodes_;
        std::shared_ptr<source::Source> source_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> potential_;
        std::vector<std::shared_ptr<Matrix<dtype::real>>> excitation_;
        std::vector<std::shared_ptr<SparseMatrix>> system_matrices_;
        std::shared_ptr<SparseMatrix> s_matrix_;
        std::shared_ptr<SparseMatrix> r_matrix_;
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix_;
        std::shared_ptr<Matrix<dtype::real>> excitation_matrix_;
        dtype::real sigma_ref_;
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
