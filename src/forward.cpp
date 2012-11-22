// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create forward_solver
template
<
    class BasisFunction,
    class NumericSolver
>
ForwardSolver<BasisFunction, NumericSolver>::ForwardSolver(Mesh* mesh, Electrodes* electrodes,
    Matrix<dtype::real>* measurmentPattern, Matrix<dtype::real>* drivePattern,
    dtype::size measurmentCount, dtype::size driveCount, dtype::size numHarmonics,
    dtype::real sigmaRef, cublasHandle_t handle, cudaStream_t stream)
    : mModel(NULL), mNumericSolver(NULL), mDriveCount(driveCount), mMeasurmentCount(measurmentCount),
        mJacobian(NULL), mVoltage(NULL), mPhi(NULL), mExcitation(NULL), mVoltageCalculation(NULL),
        mElementalJacobianMatrix(NULL) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: electrodes == NULL");
    }
    if (measurmentPattern == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: measurmentPattern == NULL");
    }
    if (drivePattern == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: drivePattern == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: handle == NULL");
    }

    // create model
    this->mModel = new Model<BasisFunction>(mesh, electrodes, sigmaRef, numHarmonics, handle,
        stream);

    // create NumericSolver solver
    this->mNumericSolver = new NumericSolver(mesh->nodeCount(), driveCount + measurmentCount, stream);

    // create matrices
    this->mJacobian = new Matrix<dtype::real>(measurmentPattern->columns() * drivePattern->columns(),
        mesh->elementCount(), stream);
    this->mVoltage  = new Matrix<dtype::real>(measurmentCount, driveCount, stream);
    this->mVoltageCalculation  = new Matrix<dtype::real>(measurmentCount, mesh->nodeCount(), stream);
    this->mElementalJacobianMatrix  = new Matrix<dtype::real>(mesh->elementCount(),
        Matrix<dtype::real>::blockSize, stream);

    // create matrix buffer
    this->mPhi = new Matrix<dtype::real>*[numHarmonics + 1];
    this->mExcitation = new Matrix<dtype::real>*[numHarmonics + 1];

    // create matrices
    for (dtype::index i = 0; i < numHarmonics + 1; i++) {
        this->mPhi[i] = new Matrix<dtype::real>(mesh->nodeCount(),
            driveCount + measurmentCount, stream);
        this->mExcitation[i] = new Matrix<dtype::real>(mesh->nodeCount(),
            driveCount + measurmentCount, stream);
    }

    // create pattern matrix
    Matrix<dtype::real> pattern(drivePattern->rows(), driveCount + measurmentCount, stream);

    // fill pattern matrix with drive pattern
    dtype::real value = 0.0f;
    for (dtype::index i = 0; i < pattern.rows(); i++) {
        for (dtype::index j = 0; j < driveCount; j++) {
            pattern(i, j) = (*drivePattern)(i, j);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::index i = 0; i < pattern.rows(); i++) {
        for (dtype::index j = 0; j < measurmentCount; j++) {
            pattern(i, j + driveCount) = (*measurmentPattern)(i, j);
        }
    }
    pattern.copyToDevice(stream);

    // calc excitation components
    this->model()->calcExcitationComponents(this->mExcitation, &pattern, handle, stream);

    // calc voltage calculation matrix
    dtype::real alpha = -1.0f, beta = 0.0f;

    // one prerun for cublas
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns(),
        this->model()->excitationMatrix()->rows(), measurmentPattern->rows(), &alpha,
        measurmentPattern->deviceData(), measurmentPattern->rows(),
        this->model()->excitationMatrix()->deviceData(), this->model()->excitationMatrix()->rows(),
        &beta, this->voltageCalculation()->deviceData(), this->voltageCalculation()->rows());

    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns(),
        this->model()->excitationMatrix()->rows(), measurmentPattern->rows(), &alpha,
        measurmentPattern->deviceData(), measurmentPattern->rows(),
        this->model()->excitationMatrix()->deviceData(), this->model()->excitationMatrix()->rows(),
        &beta, this->voltageCalculation()->deviceData(), this->voltageCalculation()->rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("ForwardSolver::ForwardSolver: calc voltage calculation");
    }

    // init jacobian calculation matrix
    this->initJacobianCalculationMatrix(handle, stream);
}

// release solver
template
<
    class BasisFunction,
    class NumericSolver
>
ForwardSolver<BasisFunction, NumericSolver>::~ForwardSolver() {
    // cleanup
    delete this->mJacobian;
    delete this->mVoltage;
    delete this->mVoltageCalculation;
    delete this->mElementalJacobianMatrix;

    if (this->mPhi != NULL) {
        for (dtype::index i = 0; i < this->model()->numHarmonics() + 1; i++) {
            delete this->mPhi[i];
        }
        delete [] this->mPhi;
    }
    if (this->mExcitation != NULL) {
        for (dtype::index i = 0; i < this->model()->numHarmonics() + 1; i++) {
            delete this->mExcitation[i];
        }
        delete [] this->mExcitation;
    }
    if (this->mModel != NULL) {
        delete this->mModel;
    }
    if (this->mNumericSolver != NULL) {
        delete this->mNumericSolver;
    }
}

// init jacobian calculation matrix
template
<
    class BasisFunction,
    class NumericSolver
>
void ForwardSolver<BasisFunction, NumericSolver>::initJacobianCalculationMatrix(cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver::initJacobianCalculationMatrix: handle == NULL");
    }

    // variables
    dtype::index id[BasisFunction::nodesPerElement];
    dtype::real x[BasisFunction::nodesPerElement * 2], y[BasisFunction::nodesPerElement * 2];
    BasisFunction* basis[BasisFunction::nodesPerElement];

    // fill connectivity and elementalJacobianMatrix
    for (dtype::index k = 0; k < this->model()->mesh()->elementCount(); k++) {
        // get nodes for element
        for (dtype::index i = 0; i < BasisFunction::nodesPerElement; i++) {
            id[i] = (*this->model()->mesh()->elements())(k, i);
            x[i] = (*this->model()->mesh()->nodes())(id[i], 0);
            y[i] = (*this->model()->mesh()->nodes())(id[i], 1);

            // get coordinates once more for permutations
            x[i + BasisFunction::nodesPerElement] = x[i];
            y[i + BasisFunction::nodesPerElement] = y[i];
        }

        // calc basis functions
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            basis[i] = new BasisFunction(&x[i], &y[i]);
        }

        // fill matrix
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            for (dtype::size j = 0; j < BasisFunction::nodesPerElement; j++) {
                // set elementalJacobianMatrix element
                (*this->mElementalJacobianMatrix)(k, i + j * BasisFunction::nodesPerElement) =
                    basis[i]->integrate_gradient_with_basis(*basis[j]);
            }
        }

        // cleanup
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            delete basis[i];
        }
    }

    // upload to device
    this->mElementalJacobianMatrix->copyToDevice(stream);
}

// forward solving
template
<
    class BasisFunction,
    class NumericSolver
>
Matrix<dtype::real>* ForwardSolver<BasisFunction, NumericSolver>::solve(Matrix<dtype::real>* gamma, dtype::size steps,
    cublasHandle_t handle, cudaStream_t stream) const {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("ForwardSolver::solve: gamma == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver::solve: handle == NULL");
    }

    // update system matrix
    this->model()->update(gamma, handle, stream);

    // solve for ground mode
    this->numericSolver()->solve(this->model()->systemMatrix(0), this->phi(0), this->excitation(0),
        steps, true, stream);

    // solve for higher harmonics
    for (dtype::index n = 1; n < this->model()->numHarmonics() + 1; n++) {
        this->numericSolver()->solve(this->model()->systemMatrix(n), this->phi(n), this->excitation(n),
            steps, false, stream);
    }

    // calc jacobian
    this->calcJacobian(gamma, 0, false, stream);
    for (dtype::index n = 1; n < this->model()->numHarmonics() + 1; n++) {
        this->calcJacobian(gamma, n, true, stream);
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    dtype::real alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation()->rows(),
        this->driveCount(), this->voltageCalculation()->columns(), &alpha,
        this->voltageCalculation()->deviceData(), this->voltageCalculation()->rows(),
        this->phi(0)->deviceData(), this->phi(0)->rows(), &beta,
        this->voltage()->deviceData(), this->voltage()->rows());

    // add harmonic voltages
    beta = 1.0f;
    for (dtype::index n = 1; n < this->model()->numHarmonics() + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation()->rows(),
            this->driveCount(), this->voltageCalculation()->columns(), &alpha,
            this->voltageCalculation()->deviceData(), this->voltageCalculation()->rows(),
            this->phi(n)->deviceData(), this->phi(n)->rows(), &beta,
            this->voltage()->deviceData(), this->voltage()->rows());
    }

    return this->voltage();
}

// specialisation
template class fastEIT::ForwardSolver<fastEIT::LinearBasis, fastEIT::SparseConjugate>;
