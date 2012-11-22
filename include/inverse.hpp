// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INVERSE_SOLVER_HPP
#define FASTEIT_INVERSE_SOLVER_HPP

// inverse solver class definition
template <class NumericSolver>
class InverseSolver {
// constructor and destructor
public:
    InverseSolver(dtype::size elementCount, dtype::size voltageCount,
        dtype::real regularizationFactor, cublasHandle_t handle, cudaStream_t stream=NULL);
    virtual ~InverseSolver();

public:
    // calc system matrix
    Matrix<dtype::real>& calcSystemMatrix(Matrix<dtype::real>& jacobian, cublasHandle_t handle,
        cudaStream_t stream = NULL);

    // calc excitation
    Matrix<dtype::real>& calcExcitation(Matrix<dtype::real>& jacobian, Matrix<dtype::real>& calculatedVoltage,
        Matrix<dtype::real>& measuredVoltage, cublasHandle_t handle, cudaStream_t stream=NULL);

    // inverse solving
    Matrix<dtype::real>& solve(Matrix<dtype::real>& gamma, Matrix<dtype::real>& jacobian,
        Matrix<dtype::real>& calculatedVoltage, Matrix<dtype::real>& measuredVoltage,
        dtype::size steps, bool regularized, cublasHandle_t handle,
        cudaStream_t stream=NULL);

// accessors
public:
    NumericSolver& numericSolver() const { return *this->mNumericSolver; }
    Matrix<dtype::real>& systemMatrix() const { return *this->mSystemMatrix; }
    Matrix<dtype::real>& jacobianSquare() const { return *this->mJacobianSquare; }
    dtype::real regularizationFactor() const { return this->mRegularizationFactor; }
    dtype::real& regularizationFactor() { return this->mRegularizationFactor; }

// member
private:
    NumericSolver* mNumericSolver;
    Matrix<dtype::real>* mDVoltage;
    Matrix<dtype::real>* mZeros;
    Matrix<dtype::real>* mExcitation;
    Matrix<dtype::real>* mSystemMatrix;
    Matrix<dtype::real>* mJacobianSquare;
    dtype::real mRegularizationFactor;
};

#endif
