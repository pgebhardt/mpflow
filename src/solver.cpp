// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create solver
Solver::Solver(Mesh* mesh, Electrodes* electrodes, Matrix<dtype::real>* measurmentPattern,
    Matrix<dtype::real>* drivePattern, dtype::real numHarmonics, dtype::real sigmaRef,
    dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream)
    : mForwardSolver(NULL), mInverseSolver(NULL), mDGamma(NULL), mGamma(NULL),
        mMeasuredVoltage(NULL), mCalibrationVoltage(NULL) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("Solver::Solver: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw invalid_argument("Solver::Solver: electrodes == NULL");
    }
    if (measurmentPattern == NULL) {
        throw invalid_argument("Solver::Solver: measurmentPattern == NULL");
    }
    if (drivePattern == NULL) {
        throw invalid_argument("Solver::Solver: drivePattern == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Solver::Solver: handle == NULL");
    }

    // create solver
    this->mForwardSolver = new ForwardSolver<Basis::Linear, SparseConjugate>(mesh, electrodes,
        measurmentPattern, drivePattern, numHarmonics, sigmaRef, handle, stream);

    this->mInverseSolver = new InverseSolver<Conjugate>(mesh->elements()->rows(),
        measurmentPattern->dataColumns() * drivePattern->dataColumns(), regularizationFactor, handle, stream);

    // create matrices
    this->mDGamma = new Matrix<dtype::real>(mesh->elements()->rows(), 1, stream);
    this->mGamma = new Matrix<dtype::real>(mesh->elements()->rows(), 1, stream);
    this->mMeasuredVoltage = new Matrix<dtype::real>(this->forwardSolver()->measurmentCount(),
        this->forwardSolver()->driveCount(), stream);
    this->mCalibrationVoltage = new Matrix<dtype::real>(this->forwardSolver()->measurmentCount(),
        this->forwardSolver()->driveCount(), stream);
}

// release solver
Solver::~Solver() {
    // cleanup
    delete this->mForwardSolver;
    delete this->mInverseSolver;
    delete this->mDGamma;
    delete this->mGamma;
    delete this->mMeasuredVoltage;
    delete this->mCalibrationVoltage;
}

// pre solve for accurate initial jacobian
Matrix<dtype::real>* Solver::preSolve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::pre_solve: handle == NULL");
    }

    // forward solving a few steps
    this->forwardSolver()->solve(this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverseSolver()->calcSystemMatrix(this->forwardSolver()->jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    this->measuredVoltage()->copy(this->forwardSolver()->voltage(), stream);
    this->calibrationVoltage()->copy(this->forwardSolver()->voltage(), stream);

    return this->forwardSolver()->voltage();
}

// calibrate
Matrix<dtype::real>* Solver::calibrate(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::calibrate: handle == NULL");
    }

    // solve forward
    this->forwardSolver()->solve(this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverseSolver()->calcSystemMatrix(this->forwardSolver()->jacobian(), handle, stream);

    // solve inverse
    this->inverseSolver()->solve(this->dGamma(), this->forwardSolver()->jacobian(),
        this->forwardSolver()->voltage(), this->calibrationVoltage(), 90, true, handle, stream);

    // add to gamma
    this->gamma()->add(this->dGamma(), stream);

    return this->gamma();
}

// solving
Matrix<dtype::real>* Solver::solve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::solve: handle == NULL");
    }

    // solve
    this->inverseSolver()->solve(this->dGamma(), this->forwardSolver()->jacobian(),
        this->calibrationVoltage(), this->measuredVoltage(), 90, false, handle, stream);

    return this->dGamma();
}
