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

    // variables
    linalgcuMatrixData_t id[BasisFunction::nodesPerElement],
        x[BasisFunction::nodesPerElement * 2], y[BasisFunction::nodesPerElement * 2];
    BasisFunction* basis[BasisFunction::nodesPerElement];

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
}

// forward solving
template <class BasisFunction>
linalgcuMatrix_t ForwardSolver<BasisFunction>::solve(linalgcuMatrix_t gamma, linalgcuSize_t steps,
    cublasHandle_t handle, cudaStream_t stream) const {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::solve: gamma == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::solve: handle == NULL");
    }

    // update system matrix
    this->model()->update(gamma, handle, stream);

    // solve for ground mode
    this->conjugateSolver()->solve(this->model()->systemMatrix(0), this->phi(0), this->excitation(0),
        steps, true, stream);

    // solve for higher harmonics
    for (linalgcuSize_t n = 1; n < this->model()->numHarmonics() + 1; n++) {
        this->conjugateSolver()->solve(this->model()->systemMatrix(n), this->phi(n), this->excitation(n),
            steps, true, stream);
    }

    // calc jacobian
    this->calc_jacobian(gamma, 0, false, stream);
    for (linalgcuSize_t n = 1; n < this->model()->numHarmonics() + 1; n++) {
        this->calc_jacobian(gamma, n, true, stream);
    }

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    linalgcuMatrixData_t alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation()->rows,
        this->driveCount(), this->voltageCalculation()->columns, &alpha,
        this->voltageCalculation()->deviceData, this->voltageCalculation()->rows,
        this->phi(0)->deviceData, this->phi(0)->rows, &beta,
        this->voltage()->deviceData, this->voltage()->rows);

    // add harmonic voltages
    beta = 1.0f;
    for (linalgcuSize_t n = 1; n < this->model()->numHarmonics() + 1; n++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltageCalculation()->rows,
            this->driveCount(), this->voltageCalculation()->columns, &alpha,
            this->voltageCalculation()->deviceData, this->voltageCalculation()->rows,
            this->phi(n)->deviceData, this->phi(n)->rows, &beta,
            this->voltage()->deviceData, this->voltage()->rows);
    }

    return this->voltage();
}

// calc jacobian kernel
template <class BasisFunction>
__global__ void calc_jacobian_kernel(linalgcuMatrixData_t* jacobian,
    linalgcuMatrixData_t* drivePhi,
    linalgcuMatrixData_t* measurmentPhi,
    linalgcuMatrixData_t* connectivityMatrix,
    linalgcuMatrixData_t* elementalJacobianMatrix,
    linalgcuMatrixData_t* gamma, linalgcuMatrixData_t sigmaRef,
    linalgcuSize_t rows, linalgcuSize_t columns,
    linalgcuSize_t phiRows, linalgcuSize_t elementCount,
    linalgcuSize_t driveCount, linalgcuSize_t measurmentCount, bool additiv) {
    // get id
    linalgcuSize_t row = blockIdx.x * blockDim.x + threadIdx.x;
    linalgcuSize_t column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    linalgcuSize_t roundMeasurmentCount = ((measurmentCount + LINALGCU_BLOCK_SIZE - 1) /
        LINALGCU_BLOCK_SIZE) * LINALGCU_BLOCK_SIZE;
    linalgcuSize_t measurmentId = row % roundMeasurmentCount;
    linalgcuSize_t driveId = row / roundMeasurmentCount;

    // variables
    linalgcuMatrixData_t dPhi[BasisFunction::nodesPerElement];
    linalgcuMatrixData_t mPhi[BasisFunction::nodesPerElement];
    linalgcuMatrixData_t id;

    // get data
    for (int i = 0; i < BasisFunction::nodesPerElement; i++) {
        id = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[(linalgcuSize_t)id + driveId * phiRows] : 0.0f;
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[(linalgcuSize_t)id +
            measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    linalgcuMatrixData_t element = 0.0f;
    for (int i = 0; i < BasisFunction::nodesPerElement; i++) {
        for (int j = 0; j < BasisFunction::nodesPerElement; j++) {
            element += dPhi[i] * mPhi[j] * elementalJacobianMatrix[column +
                (i + j * BasisFunction::nodesPerElement) * columns];
        }
    }

    // diff sigma to gamma
    element *= sigmaRef * exp10f(gamma[column] / 10.0f) / 10.0f;

    // set matrix element
    if (additiv == true) {
        jacobian[row + column * rows] += -element;
    }
    else {
        jacobian[row + column * rows] = -element;
    }
}

// calc jacobian
template <class BasisFunction>
linalgcuMatrix_t ForwardSolver<BasisFunction>::calc_jacobian(linalgcuMatrix_t gamma, linalgcuSize_t harmonic, bool additiv,
    cudaStream_t stream) const {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("ForwardSolver<BasisFunction>::calc_jacobian: gamma == NULL");
    }
    if (harmonic > this->model()->numHarmonics()) {
        throw invalid_argument("ForwardSolver<BasisFunction>::calc_jacobian: harmonic > this->model()->numHarmonics()");
    }

    // dimension
    dim3 blocks(this->jacobian()->rows / LINALGCU_BLOCK_SIZE,
        this->jacobian()->columns / LINALGCU_BLOCK_SIZE);
    dim3 threads(LINALGCU_BLOCK_SIZE, LINALGCU_BLOCK_SIZE);

    // calc jacobian
    calc_jacobian_kernel<BasisFunction><<<blocks, threads, 0, stream>>>(
        this->jacobian()->deviceData, this->phi(harmonic)->deviceData,
        &this->phi(harmonic)->deviceData[this->driveCount() * this->phi(harmonic)->rows],
        this->model()->mesh()->elements()->deviceData, this->mElementalJacobianMatrix->deviceData,
        gamma->deviceData, this->model()->sigmaRef(), this->jacobian()->rows, this->jacobian()->columns,
        this->phi(harmonic)->rows, this->model()->mesh()->elementCount(),
        this->driveCount(), this->measurmentCount(), additiv);

    return LINALGCU_SUCCESS;
}
// specialisation
template class ForwardSolver<Basis>;
