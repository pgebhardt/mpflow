// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_SOLVER_HPP
#define FASTEIT_SOLVER_HPP

// solver class definition
class Solver {
// constructor and destructor
public:
    Solver(Mesh* mesh, Electrodes* electrodes, Matrix<dtype::real>* measurmentPattern,
        Matrix<dtype::real>* drivePattern, dtype::size measurmentCount, dtype::size driveCount,
        dtype::real numHarmonics, dtype::real sigmaRef,
        dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream);
    virtual ~Solver();

public:
    // pre solve for accurate initial jacobian
    Matrix<dtype::real>* pre_solve(cublasHandle_t handle, cudaStream_t stream);

    // calibrate
    Matrix<dtype::real>* calibrate(cublasHandle_t handle, cudaStream_t stream);

    // solving
    Matrix<dtype::real>* solve(cublasHandle_t handle, cudaStream_t stream);

// accessors
public:
    ForwardSolver<LinearBasis, SparseConjugate>* forwardSolver() const {
        return this->mForwardSolver;
    }
    InverseSolver<Conjugate>* inverseSolver() const { return this->mInverseSolver; }
    Matrix<dtype::real>* dGamma() const { return this->mDGamma; }
    Matrix<dtype::real>* gamma() const { return this->mGamma; }
    Matrix<dtype::real>* measuredVoltage() const { return this->mMeasuredVoltage; }
    Matrix<dtype::real>* calibrationVoltage() const { return this->mCalibrationVoltage; }

// member
private:
    ForwardSolver<LinearBasis, SparseConjugate>* mForwardSolver;
    InverseSolver<Conjugate>* mInverseSolver;
    Matrix<dtype::real>* mDGamma;
    Matrix<dtype::real>* mGamma;
    Matrix<dtype::real>* mMeasuredVoltage;
    Matrix<dtype::real>* mCalibrationVoltage;
};

#endif
