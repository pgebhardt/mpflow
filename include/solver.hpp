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
    Solver(Mesh* mesh, Electrodes* electrodes, linalgcuMatrix_t measurmentPattern,
        linalgcuMatrix_t drivePattern, dtype::size measurmentCount, dtype::size driveCount,
        dtype::real numHarmonics, dtype::real sigmaRef,
        dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream);
    virtual ~Solver();

public:
    // pre solve for accurate initial jacobian
    linalgcuMatrix_t pre_solve(cublasHandle_t handle, cudaStream_t stream);

    // calibrate
    linalgcuMatrix_t calibrate(cublasHandle_t handle, cudaStream_t stream);

    // solving
    linalgcuMatrix_t solve(cublasHandle_t handle, cudaStream_t stream);

// accessors
public:
    ForwardSolver<LinearBasis, SparseConjugate>* forwardSolver() const {
        return this->mForwardSolver;
    }
    InverseSolver<Conjugate>* inverseSolver() const { return this->mInverseSolver; }
    linalgcuMatrix_t dGamma() const { return this->mDGamma; }
    linalgcuMatrix_t gamma() const { return this->mGamma; }
    linalgcuMatrix_t measuredVoltage() const { return this->mMeasuredVoltage; }
    linalgcuMatrix_t calibrationVoltage() const { return this->mCalibrationVoltage; }

// member
private:
    ForwardSolver<LinearBasis, SparseConjugate>* mForwardSolver;
    InverseSolver<Conjugate>* mInverseSolver;
    linalgcuMatrix_t mDGamma;
    linalgcuMatrix_t mGamma;
    linalgcuMatrix_t mMeasuredVoltage;
    linalgcuMatrix_t mCalibrationVoltage;
};

#endif
