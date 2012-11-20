// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_FORWARD_SOLVER_HPP
#define FASTEIT_FORWARD_SOLVER_HPP

// namespace fastEIT
namespace fastEIT {
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
            linalgcuMatrix_t drivePattern, linalgcuSize_t measurmentCount, linalgcuSize_t driveCount,
            linalgcuSize_t numHarmonics, linalgcuMatrixData_t sigmaRef, cublasHandle_t handle,
            cudaStream_t stream);
        virtual ~ForwardSolver();

    public:
        // init jacobian calculation matrix
        void init_jacobian_calculation_matrix(cublasHandle_t handle, cudaStream_t stream);

        // calc jacobian
        linalgcuMatrix_t calc_jacobian(linalgcuMatrix_t gamma, linalgcuSize_t harmonic, bool additiv,
            cudaStream_t stream) const;

        // forward solving
        linalgcuMatrix_t solve(linalgcuMatrix_t gamma, linalgcuSize_t steps, cublasHandle_t handle,
            cudaStream_t stream) const;

    // accessors
    public:
        Model<BasisFunction>* model() const { return this->mModel; }
        NumericSolver* numericSolver() const { return this->mNumericSolver; }
        linalgcuSize_t driveCount() const { return this->mDriveCount; }
        linalgcuSize_t measurmentCount() const { return this->mMeasurmentCount; }
        linalgcuMatrix_t jacobian() const { return this->mJacobian; }
        linalgcuMatrix_t voltage() const { return this->mVoltage; }

        linalgcuMatrix_t phi(linalgcuSize_t id) const {
            assert(id <= this->model()->numHarmonics());
            return this->mPhi[id];
        }
        linalgcuMatrix_t excitation(linalgcuSize_t id) const {
            assert(id <= this->model()->numHarmonics());
            return this->mExcitation[id];
        }

        linalgcuMatrix_t voltageCalculation() const { return this->mVoltageCalculation; }

    // member
    private:
        Model<BasisFunction>* mModel;
        SparseConjugate* mNumericSolver;
        linalgcuSize_t mDriveCount;
        linalgcuSize_t mMeasurmentCount;
        linalgcuMatrix_t mJacobian;
        linalgcuMatrix_t mVoltage;
        linalgcuMatrix_t* mPhi;
        linalgcuMatrix_t* mExcitation;
        linalgcuMatrix_t mVoltageCalculation;
        linalgcuMatrix_t mElementalJacobianMatrix;
    };
}

#endif
