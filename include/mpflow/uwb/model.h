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

#ifndef MPFLOW_INCLDUE_UWB_MODEL_H
#define MPFLOW_INCLDUE_UWB_MODEL_H

// namespaces mpFlow::UWB::model
namespace mpFlow {
namespace UWB {
namespace model {
    // model base class
    class Base {
    public:
        // constructor
        Base(std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Windows> windows);

        // update model
        virtual void update(const std::shared_ptr<numeric::Matrix<dtype::real>>,
            const std::shared_ptr<numeric::Matrix<dtype::real>>, cudaStream_t) {
        }

        // accessors
        std::shared_ptr<numeric::IrregularMesh> mesh() { return this->_mesh; }
        std::shared_ptr<Windows> windows() { return this->_windows; }
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> systemMatrix() {
            return this->_systemMatrix;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> fieldReal() {
            return this->_fieldReal;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> fieldImaginary() {
            return this->_fieldImaginary;
        }
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> sMatrix() { return this->_sMatrix; }
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> rMatrix() { return this->_rMatrix; }
        std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix() {
            return this->_connectivityMatrix;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> elementalSMatrix() {
            return this->_elementalSMatrix;
        }
        std::shared_ptr<numeric::Matrix<dtype::real>> elementalRMatrix() {
            return this->_elementalRMatrix;
        }

    protected:
        // init methods
        virtual void init(cublasHandle_t, cudaStream_t) { }
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> initElementalMatrices(cudaStream_t) {
            return nullptr;
        }

        // member
        std::shared_ptr<numeric::IrregularMesh> _mesh;
        std::shared_ptr<Windows> _windows;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> _systemMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> _fieldReal;
        std::shared_ptr<numeric::Matrix<dtype::real>> _fieldImaginary;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> _sMatrix;
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> _rMatrix;
        std::shared_ptr<numeric::Matrix<dtype::index>> _connectivityMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> _elementalSMatrix;
        std::shared_ptr<numeric::Matrix<dtype::real>> _elementalRMatrix;
    };

    // update matrix
    void updateMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> elements,
        const std::shared_ptr<numeric::Matrix<dtype::real>> material,
        const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix, cudaStream_t stream,
        std::shared_ptr<numeric::SparseMatrix<dtype::real>> result);

    // reduce matrix
    template <
        class type
    >
    void reduceMatrix(const std::shared_ptr<numeric::Matrix<type>> intermediateMatrix,
        const std::shared_ptr<numeric::SparseMatrix<dtype::real>> shape, dtype::index offset,
        cudaStream_t stream, std::shared_ptr<numeric::Matrix<type>> matrix);
}

    // model class definition
    template <
        class basis_function_type
    >
    class Model :
    public model::Base {
    public:
        // constructor
        Model(std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Windows> windows,
            cublasHandle_t handle, cudaStream_t stream);

        // update model
        virtual void update(const std::shared_ptr<numeric::Matrix<dtype::real>> epsilonR,
            const std::shared_ptr<numeric::Matrix<dtype::real>> sigma, cudaStream_t stream);

    protected:
        // init methods
        virtual void init(cublasHandle_t handle, cudaStream_t stream);
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);
    };

}
}

#endif
