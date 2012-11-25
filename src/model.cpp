// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create solver model
template <class BasisFunction>
Model<BasisFunction>::Model(Mesh* mesh, Electrodes* electrodes, dtype::real sigmaRef,
    dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream)
    : mMesh(mesh), mElectrodes(electrodes), mSigmaRef(sigmaRef), mSystemMatrix(NULL),
        mSMatrix(NULL), mRMatrix(NULL), mExcitationMatrix(NULL), mConnectivityMatrix(NULL),
        mElementalSMatrix(NULL), mElementalRMatrix(NULL), mNumHarmonics(numHarmonics) {
    // check input
    if (mesh == NULL) {
        throw invalid_argument("Model::Model: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw invalid_argument("Model::Model: electrodes == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Model::Model: handle == NULL");
    }

    // create system matrices buffer
    this->mSystemMatrix = new SparseMatrix*[this->mNumHarmonics + 1];

    // create sparse matrices
    this->createSparseMatrices(handle, stream);

    // create matrices
    this->mExcitationMatrix = new Matrix<dtype::real>(this->mesh()->nodes()->rows(),
        this->electrodes()->count(), stream);
    this->mConnectivityMatrix = new Matrix<dtype::index>(this->mesh()->nodes()->rows(),
        SparseMatrix::blockSize * Matrix<dtype::real>::blockSize, stream);
    this->mElementalSMatrix = new Matrix<dtype::real>(this->mesh()->nodes()->rows(),
        SparseMatrix::blockSize * Matrix<dtype::real>::blockSize, stream);
    this->mElementalRMatrix = new Matrix<dtype::real>(this->mesh()->nodes()->rows(),
        SparseMatrix::blockSize * Matrix<dtype::real>::blockSize, stream);

    // init model
    this->init(handle, stream);

    // init excitaion matrix
    this->initExcitationMatrix(stream);
}

// release solver model
template <class BasisFunction>
Model<BasisFunction>::~Model() {
    // cleanup
    delete this->mMesh;
    delete this->mElectrodes;
    delete this->mSMatrix;
    delete this->mRMatrix;
    delete this->mExcitationMatrix;
    delete this->mConnectivityMatrix;
    delete this->mElementalSMatrix;
    delete this->mElementalRMatrix;

    if (this->mSystemMatrix != NULL) {
        for (dtype::size i = 0; i < this->mNumHarmonics + 1; i++) {
            delete this->mSystemMatrix[i];
        }
        delete [] this->mSystemMatrix;
    }
}


// create sparse matrices
template <class BasisFunction>
void Model<BasisFunction>::createSparseMatrices(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Model::createSparseMatrices: handle == NULL");
    }

    // calc initial system matrix
    // create matrices
    Matrix<dtype::real> systemMatrix(this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);

    // calc generate empty system matrix
    dtype::index id[BasisFunction::nodesPerElement];
    for (dtype::size k = 0; k < this->mesh()->elements()->rows(); k++) {
        // get nodes for element
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            id[i] = (*this->mesh()->elements())(k, i);
        }

        // set system matrix elements
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            for (dtype::size j = 0; j < BasisFunction::nodesPerElement; j++) {
                systemMatrix(id[i], id[j]) = 1.0f;
            }
        }
    }

    // copy matrix to device
    systemMatrix.copyToDevice(stream);

    // create sparse matrices
    this->mSMatrix = new SparseMatrix(&systemMatrix, stream);
    this->mRMatrix = new SparseMatrix(&systemMatrix, stream);

    for (dtype::size i = 0; i < this->numHarmonics() + 1; i++) {
        this->mSystemMatrix[i] = new SparseMatrix(&systemMatrix, stream);
    }
}

// init model
template <class BasisFunction>
void Model<BasisFunction>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw invalid_argument("Model::init: handle == NULL");
    }

    // create intermediate matrices
    Matrix<dtype::index> elementCount(this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    Matrix<dtype::index> connectivityMatrix(this->connectivityMatrix()->dataRows(),
        elementCount.dataColumns() * Matrix<dtype::index>::blockSize, stream);
    Matrix<dtype::real> elementalSMatrix(this->elementalSMatrix()->dataRows(),
        elementCount.dataColumns() * Matrix<dtype::real>::blockSize, stream);
    Matrix<dtype::real> elementalRMatrix(this->elementalRMatrix()->dataRows(),
        elementCount.dataColumns() * Matrix<dtype::real>::blockSize, stream);

    // init connectivityMatrix
    for (dtype::size i = 0; i < connectivityMatrix.rows(); i++) {
        for (dtype::size j = 0; j < connectivityMatrix.columns(); j++) {
            connectivityMatrix(i, j) = -1;
        }
    }
    for (dtype::size i = 0; i < this->connectivityMatrix()->rows(); i++) {
        for (dtype::size j = 0; j < this->connectivityMatrix()->columns(); j++) {
            (*this->connectivityMatrix())(i, j) =  -1;
        }
    }
    this->connectivityMatrix()->copyToDevice(stream);

    // fill intermediate connectivity and elemental matrices
    dtype::index id[BasisFunction::nodesPerElement];
    dtype::real x[BasisFunction::nodesPerElement * 2], y[BasisFunction::nodesPerElement * 2];
    BasisFunction* basis[BasisFunction::nodesPerElement];
    dtype::real temp;

    for (dtype::size k = 0; k < this->mesh()->elements()->rows(); k++) {
        // get nodes for element
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            id[i] = (*this->mesh()->elements())(k, i);
            x[i] = (*this->mesh()->nodes())(id[i], 0);
            y[i] = (*this->mesh()->nodes())(id[i], 1);

            // get coordinates once more for permutations
            x[i + BasisFunction::nodesPerElement] = x[i];
            y[i + BasisFunction::nodesPerElement] = y[i];
        }

        // calc corresponding basis functions
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            basis[i] = new BasisFunction(&x[i], &y[i]);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            for (dtype::size j = 0; j < BasisFunction::nodesPerElement; j++) {
                // get current element count
                temp = elementCount(id[i], id[j]);

                // set connectivity element
                connectivityMatrix(id[i], id[j] + connectivityMatrix.dataRows() * temp) = k;

                // set elemental system element
                elementalSMatrix(id[i], id[j] + connectivityMatrix.dataRows() * temp) =
                    basis[i]->integrateGradientWithBasis(*basis[j]);

                // set elemental residual element
                elementalRMatrix(id[i], id[j] + connectivityMatrix.dataRows() * temp) =
                    basis[i]->integrateWithBasis(*basis[j]);

                // increment element count
                elementCount(id[i], id[j])++;
            }
        }

        // cleanup
        for (dtype::size i = 0; i < BasisFunction::nodesPerElement; i++) {
            delete basis[i];
        }
    }

    // upload intermediate matrices
    connectivityMatrix.copyToDevice(stream);
    elementalSMatrix.copyToDevice(stream);
    elementalRMatrix.copyToDevice(stream);

    // reduce matrices
    this->reduceMatrix(this->connectivityMatrix(), &connectivityMatrix,
        this->SMatrix()->density(), stream);
    this->reduceMatrix(this->elementalSMatrix(), &elementalSMatrix,
        this->SMatrix()->density(), stream);
    this->reduceMatrix(this->elementalRMatrix(), &elementalRMatrix,
        this->SMatrix()->density(), stream);

    // create gamma
    Matrix<dtype::real> gamma(this->mesh()->elements()->rows(), 1, stream);

    // update matrices
    this->updateMatrix(this->SMatrix(), this->elementalSMatrix(),
        &gamma, stream);
    this->updateMatrix(this->RMatrix(), this->elementalRMatrix(),
        &gamma, stream);
}

// update model
template <class BasisFunction>
void Model<BasisFunction>::update(Matrix<dtype::real>* gamma, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (gamma == NULL) {
        throw invalid_argument("Model::init: gamma == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Model::init: handle == NULL");
    }

    // update 2d systemMatrix
    this->updateMatrix(this->SMatrix(), this->elementalSMatrix(), gamma, stream);

    // update residual matrix
    this->updateMatrix(this->RMatrix(), this->elementalRMatrix(), gamma, stream);

    // set cublas stream
    cublasSetStream(handle, stream);

    // create system matrices for all harmonics
    dtype::real alpha = 0.0f;
    for (dtype::size n = 0; n < this->numHarmonics() + 1; n++) {
        // calc alpha
        alpha = (2.0f * n * M_PI / this->mesh()->height()) *
            (2.0f * n * M_PI / this->mesh()->height());

        // init system matrix with 2d system matrix
        if (cublasScopy(handle, this->SMatrix()->dataRows() * SparseMatrix::blockSize,
            this->SMatrix()->values(), 1, this->systemMatrix(n)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw logic_error(
                "Model::update: calc system matrices for all harmonics");
        }

        // add alpha * residualMatrix
        if (cublasSaxpy(handle, this->SMatrix()->dataRows() * SparseMatrix::blockSize, &alpha,
            this->RMatrix()->values(), 1, this->systemMatrix(n)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw logic_error(
                "Model::update: calc system matrices for all harmonics");
        }
    }
}

// init exitation matrix
template <class BasisFunction>
void Model<BasisFunction>::initExcitationMatrix(cudaStream_t stream) {
    // fill exitation_matrix matrix
    dtype::index id[BasisFunction::nodesPerEdge];
    dtype::real x[BasisFunction::nodesPerEdge * 2], y[BasisFunction::nodesPerEdge * 2];

    for (dtype::size i = 0; i < this->mesh()->boundary()->rows(); i++) {
        for (dtype::size l = 0; l < this->electrodes()->count(); l++) {
            for (dtype::size k = 0; k < BasisFunction::nodesPerEdge; k++) {
                // get node id
                id[k] = (*this->mesh()->boundary())(i, k);

                // get coordinates
                x[k] = (*this->mesh()->nodes())(id[k], 0);
                y[k] = (*this->mesh()->nodes())(id[k], 1);

                // set coordinates for permutations
                x[k + BasisFunction::nodesPerEdge] = x[k];
                y[k + BasisFunction::nodesPerEdge] = y[k];
            }

            // calc elements
            for (dtype::size k = 0; k < BasisFunction::nodesPerEdge; k++) {
                // add new value
                (*this->excitationMatrix())(id[k], l) -= BasisFunction::integrateBoundaryEdge(&x[k], &y[k],
                    this->electrodes()->electrodesStart(l), this->electrodes()->electrodesEnd(l)) /
                    this->electrodes()->width();
            }
        }
    }

    // upload matrix
    this->excitationMatrix()->copyToDevice(stream);
}

// calc excitaion components
template <class BasisFunction>
void Model<BasisFunction>::calcExcitationComponents(Matrix<dtype::real>** component,
    Matrix<dtype::real>* pattern, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (component == NULL) {
        throw invalid_argument("Model::calcExcitationComponents: component == NULL");
    }
    if (pattern == NULL) {
        throw invalid_argument("Model::calcExcitationComponents: pattern == NULL");
    }
    if (handle == NULL) {
        throw invalid_argument("Model::calcExcitationComponents: handle == NULL");
    }

    // calc excitation matrices
    for (dtype::size n = 0; n < this->numHarmonics() + 1; n++) {
        // Run multiply once more to avoid cublas error
        try {
            component[n]->multiply(this->excitationMatrix(), pattern, handle, stream);
        }
        catch (exception& e) {
            component[n]->multiply(this->excitationMatrix(), pattern, handle, stream);
        }
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    component[0]->scalarMultiply(1.0f / this->mesh()->height(), stream);

    // calc harmonics
    for (dtype::size n = 1; n < this->numHarmonics() + 1; n++) {
        component[n]->scalarMultiply(
            2.0f * sin(n * M_PI * this->electrodes()->height() / this->mesh()->height()) /
            (n * M_PI * this->electrodes()->height()), stream);
    }
}

// specialisation
template class fastEIT::Model<fastEIT::Basis::Linear>;
