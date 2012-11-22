// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create solver
Solver::Solver(Mesh& mesh, Electrodes& electrodes, Matrix<dtype::real>& measurmentPattern,
    Matrix<dtype::real>& drivePattern, dtype::size measurmentCount, dtype::size driveCount,
    dtype::real numHarmonics, dtype::real sigmaRef, dtype::real regularizationFactor,
    cublasHandle_t handle, cudaStream_t stream)
    : mForwardSolver(NULL), mInverseSolver(NULL), mDGamma(NULL), mGamma(NULL),
        mMeasuredVoltage(NULL), mCalibrationVoltage(NULL) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::Solver: handle == NULL");
    }

    // create matrices
    this->mDGamma = new Matrix<dtype::real>(mesh.elementCount(), 1, stream);
    this->mGamma = new Matrix<dtype::real>(mesh.elementCount(), 1, stream);
    this->mMeasuredVoltage = new Matrix<dtype::real>(measurmentCount, driveCount,
        stream);
    this->mCalibrationVoltage = new Matrix<dtype::real>(measurmentCount, driveCount,
        stream);

    // create solver
    this->mForwardSolver = new ForwardSolver<LinearBasis, SparseConjugate>(&mesh, &electrodes,
        &measurmentPattern, &drivePattern, measurmentCount, driveCount, numHarmonics, sigmaRef,
        handle, stream);

    this->mInverseSolver = new InverseSolver<Conjugate>(mesh.elementCount(),
        measurmentPattern.columns() * drivePattern.columns(), regularizationFactor, handle, stream);
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
Matrix<dtype::real>& Solver::preSolve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::pre_solve: handle == NULL");
    }

    // forward solving a few steps
    this->forwardSolver().solve(&this->gamma(), 1000, handle, stream);

    // calc system matrix
    this->inverseSolver().calcSystemMatrix(this->forwardSolver().jacobian(), handle, stream);

    // set measuredVoltage and calibrationVoltage to calculatedVoltage
    this->measuredVoltage().copy(&this->forwardSolver().voltage(), stream);
    this->calibrationVoltage().copy(&this->forwardSolver().voltage(), stream);

    return this->forwardSolver().voltage();
}

// calibrate
Matrix<dtype::real>& Solver::calibrate(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::calibrate: handle == NULL");
    }

    // solve forward
    this->forwardSolver().solve(&this->gamma(), 20, handle, stream);

    // calc inverse system matrix
    this->inverseSolver().calcSystemMatrix(this->forwardSolver().jacobian(), handle, stream);

    // solve inverse
    this->inverseSolver().solve(this->dGamma(), this->forwardSolver().jacobian(),
        this->forwardSolver().voltage(), this->calibrationVoltage(), 90, true, handle, stream);

    // add to gamma
    this->gamma().add(&this->dGamma(), stream);

    return this->gamma();
}

// solving
Matrix<dtype::real>& Solver::solve(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Solver::solve: handle == NULL");
    }

    // solve
    this->inverseSolver().solve(this->dGamma(), this->forwardSolver().jacobian(),
        this->calibrationVoltage(), this->measuredVoltage(), 90, false, handle, stream);

    return this->dGamma();
}
