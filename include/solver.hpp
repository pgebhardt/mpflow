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
    Solver(fastEIT::Mesh* mesh, fastEIT::Electrodes* electrodes,
        fastEIT::Matrix<fastEIT::dtype::real>* measurmentPattern,
        fastEIT::Matrix<fastEIT::dtype::real>* drivePattern, fastEIT::dtype::real numHarmonics,
        fastEIT::dtype::real sigmaRef, fastEIT::dtype::real regularizationFactor,
        cublasHandle_t handle, cudaStream_t stream=NULL);
    virtual ~Solver();

public:
    // pre solve for accurate initial jacobian
    fastEIT::Matrix<fastEIT::dtype::real>* preSolve(cublasHandle_t handle, cudaStream_t stream=NULL);

    // calibrate
    fastEIT::Matrix<fastEIT::dtype::real>* calibrate(cublasHandle_t handle, cudaStream_t stream=NULL);

    // solving
    fastEIT::Matrix<fastEIT::dtype::real>* solve(cublasHandle_t handle, cudaStream_t stream=NULL);

// accessors
public:
    fastEIT::ForwardSolver<fastEIT::Basis::Linear, fastEIT::Numeric::SparseConjugate>* forwardSolver() const {
        return this->mForwardSolver;
    }
    fastEIT::InverseSolver<fastEIT::Numeric::Conjugate>* inverseSolver() const { return this->mInverseSolver; }
    fastEIT::Matrix<fastEIT::dtype::real>* dGamma() const { return this->mDGamma; }
    fastEIT::Matrix<fastEIT::dtype::real>* gamma() const { return this->mGamma; }
    fastEIT::Matrix<fastEIT::dtype::real>* measuredVoltage() const { return this->mMeasuredVoltage; }
    fastEIT::Matrix<fastEIT::dtype::real>* calibrationVoltage() const { return this->mCalibrationVoltage; }

// member
private:
    fastEIT::ForwardSolver<fastEIT::Basis::Linear, fastEIT::Numeric::SparseConjugate>* mForwardSolver;
    fastEIT::InverseSolver<fastEIT::Numeric::Conjugate>* mInverseSolver;
    fastEIT::Matrix<fastEIT::dtype::real>* mDGamma;
    fastEIT::Matrix<fastEIT::dtype::real>* mGamma;
    fastEIT::Matrix<fastEIT::dtype::real>* mMeasuredVoltage;
    fastEIT::Matrix<fastEIT::dtype::real>* mCalibrationVoltage;
};

#endif
