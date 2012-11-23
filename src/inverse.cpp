// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <iostream>
#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

template <class type>
void printMatrix(fastEIT::Matrix<type>& matrix) {
    for (dtype::index i = 0; i < matrix.rows(); i++) {
        for (dtype::index j = 0; j < matrix.columns(); j++) {
            cout << matrix(i, j) << ", ";
        }
        cout << endl;
    }
}

// create inverse_solver
template <class NumericSolver>
InverseSolver<NumericSolver>::InverseSolver(dtype::size elementCount, dtype::size voltageCount,
    dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream)
    : mNumericSolver(NULL), mDVoltage(NULL), mZeros(NULL), mExcitation(NULL), mSystemMatrix(NULL),
        mJacobianSquare(NULL), mRegularizationFactor(regularizationFactor) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::InverseSolver: handle == NULL");
    }

    // create matrices
    this->mDVoltage = new Matrix<dtype::real>(voltageCount, 1, stream);
    this->mZeros = new Matrix<dtype::real>(elementCount, 1, stream);
    this->mExcitation = new Matrix<dtype::real>(elementCount, 1, stream);
    this->mSystemMatrix = new Matrix<dtype::real>(elementCount, elementCount, stream);
    this->mJacobianSquare = new Matrix<dtype::real>(elementCount, elementCount, stream);

    // create numeric solver
    this->mNumericSolver = new NumericSolver(elementCount, handle, stream);
}

// release solver
template <class NumericSolver>
InverseSolver<NumericSolver>::~InverseSolver() {
    // cleanup
    delete this->mNumericSolver;
    delete this->mDVoltage;
    delete this->mZeros;
    delete this->mExcitation;
    delete this->mSystemMatrix;
    delete this->mJacobianSquare;
}

// calc system matrix
template <class NumericSolver>
Matrix<dtype::real>* InverseSolver<NumericSolver>::calcSystemMatrix(
    Matrix<dtype::real>* jacobian, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (jacobian == NULL) {
        throw invalid_argument("InverseSolver::calcSystemMatrix: jacobian == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::calcSystemMatrix: handle == NULL");
    }

    // cublas coeficients
    dtype::real alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobianSquare()->rows(),
        this->jacobianSquare()->columns(), jacobian->rows(), &alpha, jacobian->deviceData(),
        jacobian->rows(), jacobian->deviceData(), jacobian->rows(), &beta,
        this->jacobianSquare()->deviceData(), this->jacobianSquare()->rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("InverseSolver::calcSystemMatrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    this->systemMatrix()->copy(this->jacobianSquare());

    // add lambda * Jt * J * Jt * J to systemMatrix
    beta = this->regularizationFactor();
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobianSquare()->columns(),
        this->jacobianSquare()->rows(), this->jacobianSquare()->columns(),
        &beta, this->jacobianSquare()->deviceData(),
        this->jacobianSquare()->rows(), this->jacobianSquare()->deviceData(),
        this->jacobianSquare()->rows(), &alpha, this->systemMatrix()->deviceData(),
        this->systemMatrix()->rows()) != CUBLAS_STATUS_SUCCESS) {
        throw logic_error(
            "InverseSolver::calcSystemMatrix: add lambda * Jt * J * Jt * J to systemMatrix");
    }

    return this->systemMatrix();
}

// calc excitation
template <class NumericSolver>
Matrix<dtype::real>* InverseSolver<NumericSolver>::calcExcitation(Matrix<dtype::real>* jacobian,
    Matrix<dtype::real>* calculatedVoltage, Matrix<dtype::real>* measuredVoltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == NULL) {
        throw invalid_argument("InverseSolver::calcExcitation: jacobian == NULL");
    }
    if (calculatedVoltage == NULL) {
        throw invalid_argument("InverseSolver::calcExcitation: calculatedVoltage == NULL");
    }
    if (measuredVoltage == NULL) {
        throw invalid_argument("InverseSolver::calcExcitation: measuredVoltage == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::calcExcitation: handle == NULL");
    }

    // set cublas stream
    cublasSetStream(handle, stream);

    // copy measuredVoltage to dVoltage
    if (cublasScopy(handle, this->mDVoltage->rows(),
        calculatedVoltage->deviceData(), 1, this->mDVoltage->deviceData(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error(
            "InverseSolver::calcExcitation: copy measuredVoltage to dVoltage");
    }

    // substract calculatedVoltage
    dtype::real alpha = -1.0f;
    if (cublasSaxpy(handle, this->mDVoltage->rows(), &alpha,
        measuredVoltage->deviceData(), 1, this->mDVoltage->deviceData(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error(
            "Model::calcExcitation: substract calculatedVoltage");
    }

    // calc excitation
    alpha = 1.0f;
    dtype::real beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows(), jacobian->columns(), &alpha,
        jacobian->deviceData(), jacobian->rows(), this->mDVoltage->deviceData(), 1, &beta,
        this->mExcitation->deviceData(), 1) != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("InverseSolver::calcExcitation: calc excitation");
    }

    return this->mExcitation;
}

// inverse solving
template <class NumericSolver>
Matrix<dtype::real>* InverseSolver<NumericSolver>::solve(Matrix<dtype::real>* gamma,
    Matrix<dtype::real>* jacobian, Matrix<dtype::real>* calculatedVoltage, Matrix<dtype::real>* measuredVoltage,
    dtype::size steps, bool regularized, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("InverseSolver::solve: gamma == NULL");
    }
    if (jacobian == NULL) {
        throw invalid_argument("InverseSolver::solve: jacobian == NULL");
    }
    if (measuredVoltage == NULL) {
        throw invalid_argument("InverseSolver::solve: measuredVoltage == NULL");
    }
    if (calculatedVoltage == NULL) {
        throw invalid_argument("InverseSolver::solve: calculatedVoltage == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::solve: handle == NULL");
    }

    // reset gamma
    gamma->copy(this->mZeros);

    // calc excitation
    this->calcExcitation(jacobian, calculatedVoltage, measuredVoltage, handle, stream);

    // solve system
    this->mNumericSolver->solve(regularized ? this->systemMatrix() : this->jacobianSquare(),
        gamma, this->mExcitation, steps, handle, stream);

    return gamma;
}

// specialisation
template class fastEIT::InverseSolver<fastEIT::Conjugate>;
