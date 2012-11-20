// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create solver
Solver::Solver(Mesh* mesh, Electrodes* electrodes, linalgcuMatrix_t measurmentPattern,
    linalgcuMatrix_t drivePattern, linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
    linalgcuMatrixData_t numHarmonics, linalgcuMatrixData_t sigmaRef,
    linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("Solver::Solver: mesh == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Solver::Solver: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init member
    this->mForwardSolver = NULL;
    this->mInverseSolver = NULL;
    this->mDGamma = NULL;
    this->mGamma = NULL;
    this->mMeasuredVoltage = NULL;
    this->mCalibrationVoltage = NULL;

    // create matrices
    error  = linalgcu_matrix_create(&this->mDGamma, mesh->elementCount(), 1, stream);
    error |= linalgcu_matrix_create(&this->mGamma, mesh->elementCount(), 1, stream);
    error |= linalgcu_matrix_create(&this->mMeasuredVoltage, measurmentCount, driveCount,
        stream);
    error |= linalgcu_matrix_create(&this->mCalibrationVoltage, measurmentCount, driveCount,
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("Solver::Solver: create matrices");
    }

    // create solver
    this->mForwardSolver = new ForwardSolver<LinearBasis, SparseConjugate>(mesh, electrodes,
        measurmentPattern, drivePattern, measurmentCount, driveCount, numHarmonics, sigmaRef,
        handle, stream);

    this->mInverseSolver = new InverseSolver<Conjugate>(mesh->elementCount(),
        measurmentPattern->columns * drivePattern->columns, regularizationFactor, handle, stream);
}

// release solver
Solver::~Solver() {
    // cleanup
    if (this->mForwardSolver != NULL) {
        delete this->mForwardSolver;
    }
    if (this->mInverseSolver != NULL) {
        delete this->mInverseSolver;
    }
    linalgcu_matrix_release(&this->mDGamma);
    linalgcu_matrix_release(&this->mGamma);
    linalgcu_matrix_release(&this->mMeasuredVoltage);
    linalgcu_matrix_release(&this->mCalibrationVoltage);
}

// pre solve for accurate initial jacobian
linalgcuMatrix_t Solver::pre_solve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::pre_solve: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // forward solving a few steps
    this->forwardSolver()->solve(this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverseSolver()->calc_system_matrix(this->forwardSolver()->jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    error  = linalgcu_matrix_copy(this->measuredVoltage(), this->forwardSolver()->voltage(), stream);
    error |= linalgcu_matrix_copy(this->calibrationVoltage(), this->forwardSolver()->voltage(),
        stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error(
            "Solver::pre_solve: set measuredVoltage and calibrationVoltage to calculatedVoltage");
    }

    return this->forwardSolver()->voltage();
}

// calibrate
linalgcuMatrix_t Solver::calibrate(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::calibrate: handle == NULL");
    }

    // solve forward
    this->forwardSolver()->solve(this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverseSolver()->calc_system_matrix(this->forwardSolver()->jacobian(), handle, stream);

    // solve inverse
    this->inverseSolver()->solve(this->dGamma(), this->forwardSolver()->jacobian(),
        this->forwardSolver()->voltage(), this->calibrationVoltage(), 90, true, handle, stream);

    // add to gamma
    if (linalgcu_matrix_add(this->gamma(), this->dGamma(), stream) != LINALGCU_SUCCESS) {
        throw logic_error("Solver::calibrate: add to gamma");
    }

    return this->gamma();
}

// solving
linalgcuMatrix_t Solver::solve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::solve: handle == NULL");
    }

    // solve
    this->inverseSolver()->solve(this->dGamma(), this->forwardSolver()->jacobian(),
        this->calibrationVoltage(), this->measuredVoltage(), 90, false, handle, stream);

    return this->dGamma();
}
