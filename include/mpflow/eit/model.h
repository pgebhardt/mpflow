// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLDUE_EIT_MODEL_H
#define MPFLOW_INCLDUE_EIT_MODEL_H

// namespaces mpFlow::EIT::model
namespace mpFlow {
namespace EIT {
namespace model {
    // model base class
    class Base {
    public:
        // constructor
        Base(std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Electrodes> electrodes,
            std::shared_ptr<source::Source> source,
            dtype::real sigmaRef, dtype::size component_count);

        // update model
        virtual void update(const std::shared_ptr<numeric::Matrix<dtype::real>>,
            cudaStream_t) {
        }

        // calc jacobian
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> calcJacobian(
            const std::shared_ptr<numeric::Matrix<dtype::real>>, cudaStream_t) {
            return nullptr;
        }

        // accessors
        std::shared_ptr<numeric::IrregularMesh> mesh() { return this->mesh_; }
        std::shared_ptr<Electrodes> electrodes() { return this->electrodes_; }
        std::shared_ptr<source::Source> source() {
            return this->source_;
        }
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> system_matrix(dtype::index index) {
            return this->system_matrices_[index];
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> potential(dtype::index index) {
            return this->potential_[index];
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> jacobian() { return this->jacobian_; }
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> s_matrix() { return this->s_matrix_; }
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> r_matrix() { return this->r_matrix_; }
        std::shared_ptr<numeric::Matrix<dtype::index>> connectivity_matrix() {
            return this->connectivity_matrix_;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> elemental_s_matrix() {
            return this->elemental_s_matrix_;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> elemental_r_matrix() {
            return this->elemental_r_matrix_;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> elemental_jacobian_matrix() {
            return this->elemental_jacobian_matrix_;
        }
        dtype::real sigma_ref() { return this->sigma_ref_; }
        dtype::size component_count() { return this->component_count_; }

    protected:
        // init methods
        virtual void init(cublasHandle_t, cudaStream_t) { }
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> initElementalMatrices(cudaStream_t) {
            return nullptr;
        }
        virtual void initJacobianCalculationMatrix(cublasHandle_t, cudaStream_t) {
        }

        // member
        std::shared_ptr<numeric::IrregularMesh> mesh_;
        std::shared_ptr<Electrodes> electrodes_;
        std::shared_ptr<source::Source> source_;
        std::vector<std::shared_ptr<numeric::SparseMatrix<dtype::real>>> system_matrices_;
        std::vector<std::shared_ptr<numeric::Matrix<dtype::real>>> potential_;
        std::shared_ptr<numeric::Matrix<dtype::real>> jacobian_;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> s_matrix_;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> r_matrix_;
        std::shared_ptr<numeric::Matrix<dtype::index>> connectivity_matrix_;
        std::shared_ptr<numeric::Matrix<dtype::real>> elemental_s_matrix_;
        std::shared_ptr<numeric::Matrix<dtype::real>> elemental_r_matrix_;
        std::shared_ptr<numeric::Matrix<dtype::real>> elemental_jacobian_matrix_;
        dtype::real sigma_ref_;
        dtype::size component_count_;
    };

    // update matrix
    void updateMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> elements,
        const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
        const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix, dtype::real sigmaRef,
        cudaStream_t stream, std::shared_ptr<numeric::SparseMatrix<dtype::real>> matrix);

    // reduce matrix
    template <
        class type
    >
    void reduceMatrix(const std::shared_ptr<numeric::Matrix<type>> intermediateMatrix,
        const std::shared_ptr<numeric::SparseMatrix<dtype::real>> shape, dtype::index offset,
        cudaStream_t stream, std::shared_ptr<numeric::Matrix<type>> matrix);

    // calc jacobian
    template <
        class basis_function_type
    >
    void calcJacobian(const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
        const std::shared_ptr<numeric::Matrix<dtype::real>> potential,
        const std::shared_ptr<numeric::Matrix<dtype::index>> elements,
        const std::shared_ptr<numeric::Matrix<dtype::real>> elemental_jacobian_matrix,
        dtype::size drive_count, dtype::size measurment_count,
        dtype::real sigma_ref, bool additiv,
        cudaStream_t stream, std::shared_ptr<numeric::Matrix<dtype::real>> jacobian);
}

    // model class definition
    template <
        class basis_function_type
    >
    class Model :
    public model::Base {
    public:
        // constructor
        Model(std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Electrodes> electrodes,
            std::shared_ptr<source::Source> source, dtype::real sigmaRef,
            dtype::size component_count, cublasHandle_t handle, cudaStream_t stream);

        // update model
        virtual void update(const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
            cudaStream_t stream);

        // calc jacobian
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> calcJacobian(
            const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, cudaStream_t stream);

    protected:
        // init methods
        virtual void init(cublasHandle_t handle, cudaStream_t stream);
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);
        virtual void initJacobianCalculationMatrix(cublasHandle_t handle, cudaStream_t stream);
    };

}
}

#endif
