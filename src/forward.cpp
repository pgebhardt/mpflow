// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create forward_solver
template <class BasisFunction>
ForwardSolver<BasisFunction>::ForwardSolver(Mesh* mesh, Electrodes* electrodes,
    linalgcuMatrix_t measurmentPattern, linalgcuMatrix_t drivePattern,
    linalgcuSize_t measurmentCount, linalgcuSize_t driveCount, linalgcuSize_t numHarmonics,
    linalgcuMatrixData_t sigmaRef, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::ForwardSolver: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::ForwardSolver: electrodes == NULL");
    }
    if (drivePattern == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::ForwardSolver: drivePattern == NULL");
    }
    if (measurmentPattern == NULL) {
        throw invalid_argument(
            "ForwardSolver<BasisFunction>::ForwardSolver: measurmentPattern == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::ForwardSolver: handle == NULL");
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init member
    this->mModel = NULL;
    this->mConjugateSolver = NULL;
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

    // create conjugate solver
    this->mConjugateSolver = new SparseConjugate(mesh->nodeCount(),
        this->mDriveCount + this->mMeasurmentCount, stream);

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
        throw logic_error("ForwardSolver<BasisFunction>::ForwardSolver: create matrices");
    }

    // create matrix buffer
    this->mPhi = new linalgcuMatrix_t[numHarmonics + 1];
    this->mExcitation = new linalgcuMatrix_t[numHarmonics + 1];

    // create matrices
    for (linalgcuSize_t i = 0; i < numHarmonics + 1; i++) {
        error |= linalgcu_matrix_create(&this->mPhi[i], mesh->nodeCount(),
            driveCount + measurmentCount, stream);
        error |= linalgcu_matrix_create(&this->mExcitation[i], mesh->nodeCount(),
            driveCount + measurmentCount, stream);
    }

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("ForwardSolver<BasisFunction>::ForwardSolver: create matrix buffer");
    }

    // create pattern matrix
    linalgcuMatrix_t pattern = NULL;
    error |= linalgcu_matrix_create(&pattern, drivePattern->rows,
       driveCount + measurmentCount, stream);

    // check success
    if (error != LINALGCU_SUCCESS) {
        throw logic_error("ForwardSolver<BasisFunction>::ForwardSolver: create pattern matrix");
    }

    // fill pattern matrix with drive pattern
    linalgcuMatrixData_t value = 0.0f;
    for (linalgcuSize_t i = 0; i < pattern->rows; i++) {
        for (linalgcuSize_t j = 0; j < driveCount; j++) {
            // get value
            linalgcu_matrix_get_element(drivePattern, &value, i, j);

            // set value
            linalgcu_matrix_set_element(pattern, value, i, j);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (linalgcuSize_t i = 0; i < pattern->rows; i++) {
        for (linalgcuSize_t j = 0; j < measurmentCount; j++) {
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
    linalgcuMatrixData_t alpha = -1.0f, beta = 0.0f;

    // one prerun for cublas
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        this->model()->excitationMatrix()->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        this->model()->excitationMatrix()->deviceData, this->model()->excitationMatrix()->rows,
        &beta, this->mVoltageCalculation->deviceData, this->mVoltageCalculation->rows);

    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, measurmentPattern->columns,
        this->model()->excitationMatrix()->rows, measurmentPattern->rows, &alpha,
        measurmentPattern->deviceData, measurmentPattern->rows,
        this->model()->excitationMatrix()->deviceData, this->model()->excitationMatrix()->rows,
        &beta, this->mVoltageCalculation->deviceData, this->mVoltageCalculation->rows)
        != CUBLAS_STATUS_SUCCESS) {
        throw logic_error("ForwardSolver<BasisFunction>::ForwardSolver: calc voltage calculation");
    }

    // init jacobian calculation matrix
    this->init_jacobian_calculation_matrix(handle, stream);
}

// release solver
template <class BasisFunction>
ForwardSolver<BasisFunction>::~ForwardSolver() {
    // cleanup
    linalgcu_matrix_release(&this->mJacobian);
    linalgcu_matrix_release(&this->mVoltage);
    linalgcu_matrix_release(&this->mVoltageCalculation);
    linalgcu_matrix_release(&this->mElementalJacobianMatrix);

    if (this->mPhi != NULL) {
        for (linalgcuSize_t i = 0; i < this->model()->numHarmonics() + 1; i++) {
            linalgcu_matrix_release(&this->mPhi[i]);
        }
        delete [] this->mPhi;
    }
    if (this->mExcitation != NULL) {
        for (linalgcuSize_t i = 0; i < this->model()->numHarmonics() + 1; i++) {
            linalgcu_matrix_release(&this->mExcitation[i]);
        }
        delete [] this->mExcitation;
    }
    if (this->mModel != NULL) {
        delete this->mModel;
    }
    if (this->mConjugateSolver != NULL) {
        delete this->mConjugateSolver;
    }
}

// init jacobian calculation matrix
template <class BasisFunction>
void ForwardSolver<BasisFunction>::init_jacobian_calculation_matrix(cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument(
            "ForwardSolver<BasisFunction>::init_jacobian_calculation_matrix: handle == NULL");
    }
    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // variables
    linalgcuMatrixData_t* id = new linalgcuMatrixData_t[BasisFunction::nodesPerElement];
    linalgcuMatrixData_t* x = new linalgcuMatrixData_t[BasisFunction::nodesPerElement * 2];
    linalgcuMatrixData_t* y = new linalgcuMatrixData_t[BasisFunction::nodesPerElement * 2];
    BasisFunction** basis = new BasisFunction*[BasisFunction::nodesPerElement];

    // fill connectivity and elementalJacobianMatrix
    for (linalgcuSize_t k = 0; k < this->model()->mesh()->elementCount(); k++) {
        // get nodes for element
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            linalgcu_matrix_get_element(this->model()->mesh()->elements(), &id[i], k, i);
            linalgcu_matrix_get_element(this->model()->mesh()->nodes(), &x[i],
                (linalgcuSize_t)id[i], 0);
            linalgcu_matrix_get_element(this->model()->mesh()->nodes(), &y[i],
                (linalgcuSize_t)id[i], 1);

            // get coordinates once more for permutations
            x[i + BasisFunction::nodesPerElement] = x[i];
            y[i + BasisFunction::nodesPerElement] = y[i];
        }

        // calc basis functions
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            basis[i] = new BasisFunction(&x[i], &y[i]);
        }

        // fill matrix
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            for (linalgcuSize_t j = 0; j < BasisFunction::nodesPerElement; j++) {
                // set elementalJacobianMatrix element
                linalgcu_matrix_set_element(this->mElementalJacobianMatrix,
                    basis[i]->integrate_gradient_with_basis(*basis[j]),
                    k, i + j * BasisFunction::nodesPerElement);
            }
        }

        // cleanup
        for (linalgcuSize_t i = 0; i < BasisFunction::nodesPerElement; i++) {
            delete basis[i];
        }
    }

    // upload to device
    linalgcu_matrix_copy_to_device(this->mElementalJacobianMatrix, stream);

    // cleanup
    delete [] id, x, y, basis;
}

// forward solving
template <class BasisFunction>
linalgcuMatrix_t ForwardSolver<BasisFunction>::solve(linalgcuMatrix_t gamma, linalgcuSize_t steps,
    cublasHandle_t handle, cudaStream_t stream) const {
    // check input
    if ((this == NULL) || (gamma == NULL) || (handle == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // update system matrix
    error  = fasteit_model_update(this->model, gamma, handle, stream);

    // solve for ground mode
    // solve for drive phi
    error |= fasteit_sparse_conjugate_solver_solve(this->conjugateSolver,
        this->model->systemMatrix[0], this->phi[0], this->excitation[0],
        steps, LINALGCU_TRUE, stream);

    // solve for higher harmonics
    for (linalgcuSize_t n = 1; n < this->model->numHarmonics + 1; n++) {
        // solve for drive phi
        error |= fasteit_sparse_conjugate_solver_solve(this->conjugateSolver,
            this->model->systemMatrix[n], this->phi[n], this->excitation[n],
            steps, LINALGCU_FALSE, stream);
    }

    // calc jacobian
    error |= fasteit_forward_solver_calc_jacobian(this, gamma, 0, LINALGCU_FALSE, stream);
    for (linalgcuSize_t n = 1; n < this->model->numHarmonics + 1; n++) {
        error |= fasteit_forward_solver_calc_jacobian(this, gamma, n, LINALGCU_TRUE, stream);
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation->rows,
        this->driveCount, this->voltageCalculation->columns, &alpha,
        this->voltageCalculation->deviceData, this->voltageCalculation->rows,
        this->phi[0]->deviceData, this->phi[0]->rows, &beta,
        this->voltage->deviceData, this->voltage->rows);

    // add harmonic voltages
    beta = 1.0f;
    for (linalgcuSize_t n = 1; n < this->model->numHarmonics + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation->rows,
            this->driveCount, this->voltageCalculation->columns, &alpha,
            this->voltageCalculation->deviceData, this->voltageCalculation->rows,
            this->phi[n]->deviceData, this->phi[n]->rows, &beta,
            this->voltage->deviceData, this->voltage->rows);
    }

    return error;
}

// specialisation function
void specialisation() {
    ForwardSolver<Basis> model(NULL, NULL, NULL, NULL, 0, 0, 0, 0.0f, NULL, NULL);
}
