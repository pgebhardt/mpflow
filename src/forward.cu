// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// calc jacobian kernel
template<class BasisFunction>
__global__ void calc_jacobian_kernel(dtype::real* jacobian,
    dtype::real* drivePhi,
    dtype::real* measurmentPhi,
    dtype::real* connectivityMatrix,
    dtype::real* elementalJacobianMatrix,
    dtype::real* gamma, dtype::real sigmaRef,
    dtype::size rows, dtype::size columns,
    dtype::size phiRows, dtype::size elementCount,
    dtype::size driveCount, dtype::size measurmentCount, bool additiv) {
    // get id
    dtype::size row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::size column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    dtype::size roundMeasurmentCount = ((measurmentCount + LINALGCU_BLOCK_SIZE - 1) /
        LINALGCU_BLOCK_SIZE) * LINALGCU_BLOCK_SIZE;
    dtype::size measurmentId = row % roundMeasurmentCount;
    dtype::size driveId = row / roundMeasurmentCount;

    // variables
    dtype::real dPhi[BasisFunction::nodesPerElement], mPhi[BasisFunction::nodesPerElement];
    dtype::real id;

    // get data
    for (int i = 0; i < BasisFunction::nodesPerElement; i++) {
        id = connectivityMatrix[column + i * columns];
        dPhi[i] = driveId < driveCount ? drivePhi[(dtype::size)id + driveId * phiRows] : 0.0f;
        mPhi[i] = measurmentId < measurmentCount ? measurmentPhi[(dtype::size)id +
            measurmentId * phiRows] : 0.0f;
    }

    // calc matrix element
    dtype::real element = 0.0f;
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
template
<
    class BasisFunction,
    class NumericSolver
>
linalgcuMatrix_t ForwardSolver<BasisFunction, NumericSolver>::calc_jacobian(linalgcuMatrix_t gamma,
    dtype::size harmonic, bool additiv, cudaStream_t stream) const {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("ForwardSolver::calc_jacobian: gamma == NULL");
    }
    if (harmonic > this->model()->numHarmonics()) {
        throw invalid_argument("ForwardSolver::calc_jacobian: harmonic > this->model()->numHarmonics()");
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
template class fastEIT::ForwardSolver<fastEIT::LinearBasis, fastEIT::SparseConjugate>;
