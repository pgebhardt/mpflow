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
InverseSolver<NumericSolver>::InverseSolver(linalgcuSize_t elementCount, linalgcuSize_t voltageCount,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::InverseSolver: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init member
    this->mNumericSolver = NULL;
    this->mDVoltage = NULL;
    this->mZeros = NULL;
    this->mExcitation = NULL;
    this->mSystemMatrix = NULL;
    this->mJacobianSquare = NULL;
    this->mRegularizationFactor = regularizationFactor;

    // create matrices
    error  = linalgcu_matrix_create(&this->mDVoltage, voltageCount, 1, stream);
    error |= linalgcu_matrix_create(&this->mZeros, elementCount, 1, stream);
    error |= linalgcu_matrix_create(&this->mExcitation, elementCount, 1, stream);
    error |= linalgcu_matrix_create(&this->mSystemMatrix, elementCount,
        elementCount, stream);
    error |= linalgcu_matrix_create(&this->mJacobianSquare, elementCount,
        elementCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("InverseSolver::InverseSolver: create matrices");
    }

    // create numeric solver
    this->mNumericSolver = new NumericSolver(elementCount, handle, stream);
}

// release solver
template <class NumericSolver>
InverseSolver<NumericSolver>::~InverseSolver() {
    // cleanup
    if (this->mNumericSolver != NULL) {
        delete this->mNumericSolver;
    }
    linalgcu_matrix_release(&this->mDVoltage);
    linalgcu_matrix_release(&this->mZeros);
    linalgcu_matrix_release(&this->mExcitation);
    linalgcu_matrix_release(&this->mSystemMatrix);
    linalgcu_matrix_release(&this->mJacobianSquare);
}

// calc system matrix
template <class NumericSolver>
linalgcuMatrix_t InverseSolver<NumericSolver>::calc_system_matrix(
    linalgcuMatrix_t jacobian, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (jacobian == NULL) {
        throw invalid_argument("InverseSolver::calc_system_matrix: jacobian == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::calc_system_matrix: handle == NULL");
    }

    // cublas coeficients
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;

    // calc Jt * J
    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, this->jacobianSquare()->rows,
        this->jacobianSquare()->columns, jacobian->rows, &alpha, jacobian->deviceData,
        jacobian->rows, jacobian->deviceData, jacobian->rows, &beta,
        this->jacobianSquare()->deviceData, this->jacobianSquare()->rows)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("InverseSolver::calc_system_matrix: calc Jt * J");
    }

    // copy jacobianSquare to systemMatrix
    if (linalgcu_matrix_copy(this->systemMatrix(), this->jacobianSquare(), stream)
        != LINALGCU_SUCCESS) {
        throw logic_error("InverseSolver::calc_system_matrix: copy jacobianSquare to systemMatrix");
    }

    // add lambda * Jt * J * Jt * J to systemMatrix
    if (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->jacobianSquare()->columns,
        this->jacobianSquare()->rows, this->jacobianSquare()->columns,
        &this->mRegularizationFactor, this->jacobianSquare()->deviceData,
        this->jacobianSquare()->rows, this->jacobianSquare()->deviceData,
        this->jacobianSquare()->rows, &alpha, this->systemMatrix()->deviceData,
        this->systemMatrix()->rows) != CUBLAS_STATUS_SUCCESS) {
        throw logic_error(
            "InverseSolver::calc_system_matrix: add lambda * Jt * J * Jt * J to systemMatrix");
    }

    return this->systemMatrix();
}

// calc excitation
template <class NumericSolver>
linalgcuMatrix_t InverseSolver<NumericSolver>::calc_excitation(linalgcuMatrix_t jacobian,
    linalgcuMatrix_t calculatedVoltage, linalgcuMatrix_t measuredVoltage, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (jacobian == NULL) {
        throw invalid_argument("InverseSolver::calc_excitation: jacobian == NULL");
    }
    if (calculatedVoltage == NULL) {
        throw invalid_argument("InverseSolver::calc_excitation: calculatedVoltage == NULL");
    }
    if (measuredVoltage == NULL) {
        throw invalid_argument("InverseSolver::calc_excitation: measuredVoltage == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::calc_excitation: handle == NULL");
    }

    // dummy matrix to turn matrix to column vector
    linalgcuMatrix_s dummy_matrix;
    dummy_matrix.rows = this->mDVoltage->rows;
    dummy_matrix.columns = this->mDVoltage->columns;
    dummy_matrix.hostData = NULL;

    // calc deltaVoltage = mv - cv
    dummy_matrix.deviceData = calculatedVoltage->deviceData;
    linalgcu_matrix_copy(this->mDVoltage, &dummy_matrix, stream);
    linalgcu_matrix_scalar_multiply(this->mDVoltage, -1.0f, stream);

    dummy_matrix.deviceData = measuredVoltage->deviceData;
    linalgcu_matrix_add(this->mDVoltage, &dummy_matrix, stream);

    // set cublas stream
    cublasSetStream(handle, stream);

    // calc excitation
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
        jacobian->deviceData, jacobian->rows, this->mDVoltage->deviceData, 1, &beta,
        this->mExcitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {

        // try once again
        if (cublasSgemv(handle, CUBLAS_OP_T, jacobian->rows, jacobian->columns, &alpha,
            jacobian->deviceData, jacobian->rows, this->mDVoltage->deviceData, 1, &beta,
            this->mExcitation->deviceData, 1) != CUBLAS_STATUS_SUCCESS) {
            throw logic_error("InverseSolver::calc_excitation: calc excitation");
        }
    }

    return this->mExcitation;
}

// inverse solving
template <class NumericSolver>
linalgcuMatrix_t InverseSolver<NumericSolver>::solve(linalgcuMatrix_t gamma,
    linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage, linalgcuMatrix_t measuredVoltage,
    linalgcuSize_t steps, bool regularized, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("InverseSolver::solve: gamma == NULL");
    }
    if (jacobian == NULL) {
        throw invalid_argument("InverseSolver::solve: jacobian == NULL");
    }
    if (calculatedVoltage == NULL) {
        throw invalid_argument("InverseSolver::solve: calculatedVoltage == NULL");
    }
    if (measuredVoltage == NULL) {
        throw invalid_argument("InverseSolver::solve: measuredVoltage == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("InverseSolver::solve: handle == NULL");
    }

    // reset gamma
    if (linalgcu_matrix_copy(gamma, this->mZeros, stream) != LINALGCU_SUCCESS) {
        throw logic_error("InverseSolver::solve: reset gamma");
    }

    // calc excitation
    this->calc_excitation(jacobian, calculatedVoltage, measuredVoltage, handle, stream);

    // solve system
    this->mNumericSolver->solve(regularized ? this->systemMatrix() : this->jacobianSquare(),
        gamma, this->mExcitation, steps, handle, stream);

    return gamma;
}

// specialisation
template class fastEIT::InverseSolver<fastEIT::Conjugate>;
