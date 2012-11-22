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
    InverseSolver(linalgcuSize_t elementCount, linalgcuSize_t voltageCount,
        linalgcuMatrixData_t regularizationFactor, cublasHandle_t handle, cudaStream_t stream);
    virtual ~InverseSolver();

public:
    // calc system matrix
    linalgcuMatrix_t calc_system_matrix(linalgcuMatrix_t jacobian, cublasHandle_t handle,
        cudaStream_t stream);

    // calc excitation
    linalgcuMatrix_t calc_excitation(linalgcuMatrix_t jacobian, linalgcuMatrix_t calculatedVoltage,
        linalgcuMatrix_t measuredVoltage, cublasHandle_t handle, cudaStream_t stream);

    // inverse solving
    linalgcuMatrix_t solve(linalgcuMatrix_t gamma, linalgcuMatrix_t jacobian,
        linalgcuMatrix_t calculatedVoltage, linalgcuMatrix_t measuredVoltage,
        linalgcuSize_t steps, bool regularized, cublasHandle_t handle,
        cudaStream_t stream);

// accessors
public:
    NumericSolver* numericSolver() const { return this->mNumericSolver; }
    linalgcuMatrix_t systemMatrix() const { return this->mSystemMatrix; }
    linalgcuMatrix_t jacobianSquare() const { return this->mJacobianSquare; }
    linalgcuMatrixData_t regularizationFactor() const { return this->mRegularizationFactor; }
    linalgcuMatrixData_t& regularizationFactor() { return this->mRegularizationFactor; }

// member
private:
    NumericSolver* mNumericSolver;
    linalgcuMatrix_t mDVoltage;
    linalgcuMatrix_t mZeros;
    linalgcuMatrix_t mExcitation;
    linalgcuMatrix_t mSystemMatrix;
    linalgcuMatrix_t mJacobianSquare;
    linalgcuMatrixData_t mRegularizationFactor;
};

#endif
