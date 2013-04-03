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
        Model(std::shared_ptr<Mesh> mesh, std::shared_ptr<Electrodes> electrodes,
            std::shared_ptr<source::Source<template_basis_function_type>> source,
            dtype::real sigmaRef, dtype::size components_count, cublasHandle_t handle,
            cudaStream_t stream);

        // update model
        void update(const std::shared_ptr<Matrix<dtype::real>> gamma, cublasHandle_t handle,
            cudaStream_t stream);

        // type defs
        typedef template_basis_function_type basis_function_type;

        // accessors
        std::shared_ptr<Mesh> mesh() { return this->mesh_; }
        std::shared_ptr<Electrodes> electrodes() { return this->electrodes_; }
        std::shared_ptr<source::Source<template_basis_function_type>> source() {
            return this->source_;
        }
        std::shared_ptr<SparseMatrix<dtype::real>> system_matrix(dtype::index index) { return this->system_matrices_[index]; }
        std::shared_ptr<SparseMatrix<dtype::real>> s_matrix() { return this->s_matrix_; }
        std::shared_ptr<SparseMatrix<dtype::real>> r_matrix() { return this->r_matrix_; }
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix() { return this->connectivity_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix() { return this->elemental_s_matrix_; }
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix() { return this->elemental_r_matrix_; }
        dtype::real sigma_ref() { return this->sigma_ref_; }
        dtype::size components_count() { return this->components_count_; }

    private:
        // init methods
        void init(cublasHandle_t handle, cudaStream_t stream);
        std::shared_ptr<Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);

        // member
        std::shared_ptr<Mesh> mesh_;
        std::shared_ptr<Electrodes> electrodes_;
        std::shared_ptr<source::Source<template_basis_function_type>> source_;
        std::vector<std::shared_ptr<SparseMatrix<dtype::real>>> system_matrices_;
        std::shared_ptr<SparseMatrix<dtype::real>> s_matrix_;
        std::shared_ptr<SparseMatrix<dtype::real>> r_matrix_;
        std::shared_ptr<Matrix<dtype::index>> connectivity_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_s_matrix_;
        std::shared_ptr<Matrix<dtype::real>> elemental_r_matrix_;
        dtype::real sigma_ref_;
        dtype::size components_count_;
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
    }
}

#endif
