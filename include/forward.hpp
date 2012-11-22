// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_FORWARD_SOLVER_HPP
#define FASTEIT_FORWARD_SOLVER_HPP

// forward solver class definition
template
<
    class BasisFunction,
    class NumericSolver
>
class ForwardSolver {
// constructor and destructor
public:
    ForwardSolver(Mesh* mesh, Electrodes* electrodes, Matrix<dtype::real>* measurmentPattern,
        Matrix<dtype::real>* drivePattern, dtype::size measurmentCount, dtype::size driveCount,
        dtype::size numHarmonics, dtype::real sigmaRef, cublasHandle_t handle,
        cudaStream_t stream=NULL);
    virtual ~ForwardSolver();

public:
    // init jacobian calculation matrix
    void initJacobianCalculationMatrix(cublasHandle_t handle, cudaStream_t stream=NULL);

    // calc jacobian
    Matrix<dtype::real>* calcJacobian(Matrix<dtype::real>* gamma, dtype::size harmonic, bool additiv,
        cudaStream_t stream=NULL) const;

    // forward solving
    Matrix<dtype::real>* solve(Matrix<dtype::real>* gamma, dtype::size steps, cublasHandle_t handle,
        cudaStream_t stream=NULL) const;

// accessors
public:
    Model<BasisFunction>* model() const { return this->mModel; }
    NumericSolver* numericSolver() const { return this->mNumericSolver; }
    dtype::size driveCount() const { return this->mDriveCount; }
    dtype::size measurmentCount() const { return this->mMeasurmentCount; }
    Matrix<dtype::real>* jacobian() const { return this->mJacobian; }
    Matrix<dtype::real>* voltage() const { return this->mVoltage; }

    Matrix<dtype::real>* phi(dtype::size id) const {
        assert(id <= this->model()->numHarmonics());
        return this->mPhi[id];
    }
    Matrix<dtype::real>* excitation(dtype::size id) const {
        assert(id <= this->model()->numHarmonics());
        return this->mExcitation[id];
    }

    Matrix<dtype::real>* voltageCalculation() const { return this->mVoltageCalculation; }

// member
private:
    Model<BasisFunction>* mModel;
    SparseConjugate* mNumericSolver;
    dtype::size mDriveCount;
    dtype::size mMeasurmentCount;
    Matrix<dtype::real>* mJacobian;
    Matrix<dtype::real>* mVoltage;
    Matrix<dtype::real>** mPhi;
    Matrix<dtype::real>** mExcitation;
    Matrix<dtype::real>* mVoltageCalculation;
    Matrix<dtype::real>* mElementalJacobianMatrix;
};

#endif
