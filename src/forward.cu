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
__global__ void calcJacobianKernel(dtype::real* jacobian,
    dtype::real* drivePhi,
    dtype::real* measurmentPhi,
    dtype::index* connectivityMatrix,
    dtype::real* elementalJacobianMatrix,
    dtype::real* gamma, dtype::real sigmaRef,
    dtype::size rows, dtype::size columns,
    dtype::size phiRows, dtype::size elementCount,
    dtype::size driveCount, dtype::size measurmentCount, bool additiv) {
    // get id
    dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // check column
    if (column >= elementCount) {
        return;
    }

    // calc measurment and drive id
    dtype::size roundMeasurmentCount = ((measurmentCount + Matrix<dtype::real>::blockSize - 1) /
        Matrix<dtype::real>::blockSize) * Matrix<dtype::real>::blockSize;
    dtype::size measurmentId = row % roundMeasurmentCount;
    dtype::size driveId = row / roundMeasurmentCount;

    // variables
    dtype::real dPhi[BasisFunction::nodesPerElement], mPhi[BasisFunction::nodesPerElement];
    dtype::index id;

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
Matrix<dtype::real>& ForwardSolver<BasisFunction, NumericSolver>::calcJacobian(Matrix<dtype::real>& gamma,
    dtype::size harmonic, bool additiv, cudaStream_t stream) const {
    // check input
    if (harmonic > this->model().numHarmonics()) {
        throw invalid_argument("ForwardSolver::calcJacobian: harmonic > this->model()->numHarmonics()");
    }

    // dimension
    dim3 blocks(this->jacobian().rows() / Matrix<dtype::real>::blockSize,
        this->jacobian().columns() / Matrix<dtype::real>::blockSize);
    dim3 threads(Matrix<dtype::real>::blockSize, Matrix<dtype::real>::blockSize);

    // calc jacobian
    calcJacobianKernel<BasisFunction><<<blocks, threads, 0, stream>>>(
        this->jacobian().deviceData(), this->phi(harmonic).deviceData(),
        &this->phi(harmonic).deviceData()[this->driveCount() * this->phi(harmonic).rows()],
        this->model().mesh().elements().deviceData(), this->mElementalJacobianMatrix->deviceData(),
        gamma.deviceData(), this->model().sigmaRef(), this->jacobian().rows(), this->jacobian().columns(),
        this->phi(harmonic).rows(), this->model().mesh().elementCount(),
        this->driveCount(), this->measurmentCount(), additiv);

    return this->jacobian();
}

// specialisation
template class fastEIT::ForwardSolver<fastEIT::LinearBasis, fastEIT::SparseConjugate>;
