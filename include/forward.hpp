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
    ForwardSolver(Mesh* mesh, Electrodes* electrodes, linalgcuMatrix_t measurmentPattern,
        linalgcuMatrix_t drivePattern, dtype::size measurmentCount, dtype::size driveCount,
        dtype::size numHarmonics, dtype::real sigmaRef, cublasHandle_t handle,
        cudaStream_t stream);
    virtual ~ForwardSolver();

public:
    // init jacobian calculation matrix
    void init_jacobian_calculation_matrix(cublasHandle_t handle, cudaStream_t stream);

    // calc jacobian
    linalgcuMatrix_t calc_jacobian(linalgcuMatrix_t gamma, dtype::size harmonic, bool additiv,
        cudaStream_t stream) const;

    // forward solving
    linalgcuMatrix_t solve(linalgcuMatrix_t gamma, dtype::size steps, cublasHandle_t handle,
        cudaStream_t stream) const;

// accessors
public:
    Model<BasisFunction>* model() const { return this->mModel; }
    NumericSolver* numericSolver() const { return this->mNumericSolver; }
    dtype::size driveCount() const { return this->mDriveCount; }
    dtype::size measurmentCount() const { return this->mMeasurmentCount; }
    linalgcuMatrix_t jacobian() const { return this->mJacobian; }
    linalgcuMatrix_t voltage() const { return this->mVoltage; }

    linalgcuMatrix_t phi(dtype::size id) const {
        assert(id <= this->model()->numHarmonics());
        return this->mPhi[id];
    }
    linalgcuMatrix_t excitation(dtype::size id) const {
        assert(id <= this->model()->numHarmonics());
        return this->mExcitation[id];
    }

    linalgcuMatrix_t voltageCalculation() const { return this->mVoltageCalculation; }

// member
private:
    Model<BasisFunction>* mModel;
    SparseConjugate* mNumericSolver;
    dtype::size mDriveCount;
    dtype::size mMeasurmentCount;
    linalgcuMatrix_t mJacobian;
    linalgcuMatrix_t mVoltage;
    linalgcuMatrix_t* mPhi;
    linalgcuMatrix_t* mExcitation;
    linalgcuMatrix_t mVoltageCalculation;
    linalgcuMatrix_t mElementalJacobianMatrix;
};

#endif
