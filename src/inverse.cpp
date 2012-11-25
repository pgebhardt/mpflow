// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

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
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobianSquare()->dataRows(),
        this->jacobianSquare()->dataColumns(), jacobian->dataRows(), &alpha, jacobian->deviceData(),
        jacobian->dataRows(), jacobian->deviceData(), jacobian->dataRows(), &beta,
        this->jacobianSquare()->deviceData(), this->jacobianSquare()->dataRows())
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("InverseSolver::calcSystemMatrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    this->systemMatrix()->copy(this->jacobianSquare());

    // add lambda * Jt * J * Jt * J to systemMatrix
    beta = this->regularizationFactor();
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobianSquare()->dataColumns(),
        this->jacobianSquare()->dataRows(), this->jacobianSquare()->dataColumns(),
        &beta, this->jacobianSquare()->deviceData(),
        this->jacobianSquare()->dataRows(), this->jacobianSquare()->deviceData(),
        this->jacobianSquare()->dataRows(), &alpha, this->systemMatrix()->deviceData(),
        this->systemMatrix()->dataRows()) != CUBLAS_STATUS_SUCCESS) {
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
    if (cublasScopy(handle, this->dVoltage()->dataRows(),
        calculatedVoltage->deviceData(), 1, this->dVoltage()->deviceData(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error(
            "InverseSolver::calcExcitation: copy measuredVoltage to dVoltage");
    }

    // substract calculatedVoltage
    dtype::real alpha = -1.0f;
    if (cublasSaxpy(handle, this->dVoltage()->dataRows(), &alpha,
        measuredVoltage->deviceData(), 1, this->dVoltage()->deviceData(), 1)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error(
            "Model::calcExcitation: substract calculatedVoltage");
    }

    // calc excitation
    alpha = 1.0f;
    dtype::real beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->dataRows(), jacobian->dataColumns(), &alpha,
        jacobian->deviceData(), jacobian->dataRows(), this->dVoltage()->deviceData(), 1, &beta,
        this->excitation()->deviceData(), 1) != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("InverseSolver::calcExcitation: calc excitation");
    }

    return this->excitation();
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
    gamma->copy(this->zeros());

    // calc excitation
    this->calcExcitation(jacobian, calculatedVoltage, measuredVoltage, handle, stream);

    // solve system
    this->numericSolver()->solve(regularized ? this->systemMatrix() : this->jacobianSquare(),
        gamma, this->excitation(), steps, handle, stream);

    return gamma;
}

// specialisation
template class fastEIT::InverseSolver<fastEIT::Numeric::Conjugate>;
