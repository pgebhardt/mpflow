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
    linalgcuMatrix_t measurmentPattern, linalgcuMatrix_t drivePattern,
    dtype::size measurmentCount, dtype::size driveCount, dtype::size numHarmonics,
    dtype::real sigmaRef, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: electrodes == NULL");
    }
    if (drivePattern == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: drivePattern == NULL");
    }
    if (measurmentPattern == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: measurmentPattern == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver::ForwardSolver: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init member
    this->mModel = NULL;
    this->mNumericSolver = NULL;
    this->mDriveCount = driveCount;
    this->mMeasurmentCount = measurmentCount;
    this->mJacobian = NULL;
    this->mVoltage = NULL;
    this->mPhi = NULL;
    this->mExcitation = NULL;
    this->mVoltageCalculation = NULL;
    this->mElementalJacobianMatrix = NULL;

    // create model
    this->mModel = new Model<BasisFunction>(mesh, electrodes, sigmaRef, numHarmonics, handle,
        stream);

    // create NumericSolver solver
    this->mNumericSolver = new NumericSolver(mesh->nodeCount(),
        driveCount + measurmentCount, stream);

    // create matrices
    error  = linalgcu_matrix_create(&this->mJacobian,
        measurmentPattern->columns * drivePattern->columns, mesh->elementCount(), stream);
    error |= linalgcu_matrix_create(&this->mVoltage, measurmentCount, driveCount, stream);
    error |= linalgcu_matrix_create(&this->mVoltageCalculation, measurmentCount,
        mesh->nodeCount(), stream);
    error |= linalgcu_matrix_create(&this->mElementalJacobianMatrix, mesh->elementCount(),
        LINALGCU_BLOCK_SIZE, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("ForwardSolver::ForwardSolver: create matrices");
    }

    // create matrix buffer
    this->mPhi = new linalgcuMatrix_t[numHarmonics + 1];
    this->mExcitation = new linalgcuMatrix_t[numHarmonics + 1];

    // create matrices
    for (dtype::size i = 0; i < numHarmonics + 1; i++) {
        error |= linalgcu_matrix_create(&this->mPhi[i], mesh->nodeCount(),
            driveCount + measurmentCount, stream);
        error |= linalgcu_matrix_create(&this->mExcitation[i], mesh->nodeCount(),
            driveCount + measurmentCount, stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("ForwardSolver::ForwardSolver: create matrix buffer");
    }

    // create pattern matrix
    linalgcuMatrix_t pattern = NULL;
    error |= linalgcu_matrix_create(&pattern, drivePattern->rows,
       driveCount + measurmentCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("ForwardSolver::ForwardSolver: create pattern matrix");
    }

    // fill pattern matrix with drive pattern
    dtype::real value = 0.0f;
    for (dtype::size i = 0; i < pattern->rows; i++) {
        for (dtype::size j = 0; j < driveCount; j++) {
            // get value
            linalgcu_matrix_get_element(drivePattern, &value, i, j);

            // set value
            linalgcu_matrix_set_element(pattern, value, i, j);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::size i = 0; i < pattern->rows; i++) {
        for (dtype::size j = 0; j < measurmentCount; j++) {
            // get value
            linalgcu_matrix_get_element(measurmentPattern, &value, i, j);

            // set value
            linalgcu_matrix_set_element(pattern, -value, i, j + driveCount);
        }
    }

    linalgcu_matrix_copy_to_device(pattern, stream);

    // calc excitation components
    this->model()->calc_excitation_components(this->mExcitation, pattern, handle, stream);

    // cleanup
    linalgcu_matrix_release(&pattern);

    // calc voltage calculation matrix
    dtype::real alpha = -1.0f, beta = 0.0f;

    // one prerun for cublas
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        this->model()->excitationMatrix()->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        this->model()->excitationMatrix()->deviceData, this->model()->excitationMatrix()->rows,
        &beta, this->voltageCalculation()->deviceData, this->voltageCalculation()->rows);

    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        this->model()->excitationMatrix()->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        this->model()->excitationMatrix()->deviceData, this->model()->excitationMatrix()->rows,
        &beta, this->voltageCalculation()->deviceData, this->voltageCalculation()->rows)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("ForwardSolver::ForwardSolver: calc voltage calculation");
    }

    // init jacobian calculation matrix
    this->init_jacobian_calculation_matrix(handle, stream);
}

// release solver
template
<
    class BasisFunction,
    class NumericSolver
>
ForwardSolver<BasisFunction, NumericSolver>::~ForwardSolver() {
    // cleanup
    linalgcu_matrix_release(&this->mJacobian);
    linalgcu_matrix_release(&this->mVoltage);
    linalgcu_matrix_release(&this->mVoltageCalculation);
    linalgcu_matrix_release(&this->mElementalJacobianMatrix);

    if (this->mPhi != NULL) {
        for (dtype::size i = 0; i < this->model()->numHarmonics() + 1; i++) {
            linalgcu_matrix_release(&this->mPhi[i]);
        }
        delete [] this->mPhi;
    }
    if (this->mExcitation != NULL) {
        for (dtype::size i = 0; i < this->model()->numHarmonics() + 1; i++) {
            linalgcu_matrix_release(&this->mExcitation[i]);
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
void ForwardSolver<BasisFunction, NumericSolver>::init_jacobian_calculation_matrix(cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver::init_jacobian_calculation_matrix: handle == NULL");
    }

    // variables
    dtype::real id[BasisFunction::nodesPerElement],
        x[BasisFunction::nodesPerElement * 2], y[BasisFunction::nodesPerElement * 2];
    BasisFunction* basis[BasisFunction::nodesPerElement];

    // fill connectivity and elementalJacobianMatrix
    for (dtype::size k = 0; k < this->model()->mesh()->elementCount(); k++) {
        // get nodes for element
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            linalgcu_matrix_get_element(this->model()->mesh()->elements(), &id[i], k, i);
            linalgcu_matrix_get_element(this->model()->mesh()->nodes(), &x[i],
                (dtype::size)id[i], 0);
            linalgcu_matrix_get_element(this->model()->mesh()->nodes(), &y[i],
                (dtype::size)id[i], 1);

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
                linalgcu_matrix_set_element(this->mElementalJacobianMatrix,
                    basis[i]->integrate_gradient_with_basis(*basis[j]),
                    k, i + j * BasisFunction::nodesPerElement);
            }
        }

        // cleanup
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            delete basis[i];
        }
    }

    // upload to device
    linalgcu_matrix_copy_to_device(this->mElementalJacobianMatrix, stream);
}

// forward solving
template
<
    class BasisFunction,
    class NumericSolver
>
linalgcuMatrix_t ForwardSolver<BasisFunction, NumericSolver>::solve(linalgcuMatrix_t gamma, dtype::size steps,
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
    for (dtype::size n = 1; n < this->model()->numHarmonics() + 1; n++) {
        this->numericSolver()->solve(this->model()->systemMatrix(n), this->phi(n), this->excitation(n),
            steps, false, stream);
    }

    // calc jacobian
    this->calc_jacobian(gamma, 0, false, stream);
    for (dtype::size n = 1; n < this->model()->numHarmonics() + 1; n++) {
        this->calc_jacobian(gamma, n, true, stream);
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    dtype::real alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation()->rows,
        this->driveCount(), this->voltageCalculation()->columns, &alpha,
        this->voltageCalculation()->deviceData, this->voltageCalculation()->rows,
        this->phi(0)->deviceData, this->phi(0)->rows, &beta,
        this->voltage()->deviceData, this->voltage()->rows);

    // add harmonic voltages
    beta = 1.0f;
    for (dtype::size n = 1; n < this->model()->numHarmonics() + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation()->rows,
            this->driveCount(), this->voltageCalculation()->columns, &alpha,
            this->voltageCalculation()->deviceData, this->voltageCalculation()->rows,
            this->phi(n)->deviceData, this->phi(n)->rows, &beta,
            this->voltage()->deviceData, this->voltage()->rows);
    }

    return this->voltage();
}

// specialisation
template class fastEIT::ForwardSolver<fastEIT::LinearBasis, fastEIT::SparseConjugate>;
