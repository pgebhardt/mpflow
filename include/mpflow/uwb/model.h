// mpFlow
//
// Copyright (C) 2014  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

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
        virtual void update(const std::shared_ptr<numeric::Matrix<dtype::real>> realPart,
            const std::shared_ptr<numeric::Matrix<dtype::real>> imaginaryPart, cudaStream_t stream);

    protected:
        // init methods
        virtual void init(cublasHandle_t handle, cudaStream_t stream);
        virtual std::shared_ptr<numeric::Matrix<dtype::real>> initElementalMatrices(cudaStream_t stream);
    };

}
}

#endif
