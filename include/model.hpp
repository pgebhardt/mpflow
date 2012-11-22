// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_MODEL_HPP
#define FASTEIT_MODEL_HPP

// model class definition
template <class BasisFunction>
class Model {
// constructor and destructor
public:
    Model(Mesh* mesh, Electrodes* electrodes, dtype::real sigmaRef,
        dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream=NULL);
    virtual ~Model();

// init methods
private:
    void init(cublasHandle_t handle, cudaStream_t stream=NULL);
    void create_sparse_matrices(cublasHandle_t handle, cudaStream_t stream=NULL);
    void init_excitation_matrix(cudaStream_t stream=NULL);

public:
    // calc excitaion components
    void calc_excitation_components(Matrix<dtype::real>** component, Matrix<dtype::real>* pattern,
        cublasHandle_t handle, cudaStream_t stream=NULL);

    // update model
    void update(Matrix<dtype::real>* gamma, cublasHandle_t handle, cudaStream_t stream=NULL);

// cuda methods
private:
    // update matrix
    void update_matrix(SparseMatrix* matrix, Matrix<dtype::real>* elements,
        Matrix<dtype::real>* gamma, cudaStream_t stream=NULL);

    // reduce matrix
    void reduce_matrix(Matrix<dtype::real>* matrix, Matrix<dtype::real>* intermediateMatrix,
        dtype::size density, cudaStream_t stream=NULL);
    void reduce_matrix(Matrix<dtype::index>* matrix, Matrix<dtype::index>* intermediateMatrix,
        dtype::size density, cudaStream_t stream=NULL);

// access methods
public:
    Mesh& mesh() const { return *this->mMesh; }
    Electrodes& electrodes() const { return *this->mElectrodes; }
    dtype::real sigmaRef() const { return this->mSigmaRef; }
    inline SparseMatrix& systemMatrix(dtype::size id) {
        assert(id <= this->mNumHarmonics);
        return *this->mSystemMatrix[id];
    }
    Matrix<dtype::real>& excitationMatrix() { return *this->mExcitationMatrix; }
    dtype::size numHarmonics() { return this->mNumHarmonics; }

// geometry definition
public:
    static const dtype::size nodesPerEdge = BasisFunction::nodesPerElement;
    static const dtype::size nodesPerElement = BasisFunction::nodesPerEdge;

// member
private:
    Mesh* mMesh;
    Electrodes* mElectrodes;
    dtype::real mSigmaRef;
    SparseMatrix** mSystemMatrix;
    SparseMatrix* mSMatrix;
    SparseMatrix* mRMatrix;
    Matrix<dtype::real>* mExcitationMatrix;
    Matrix<dtype::index>* mConnectivityMatrix;
    Matrix<dtype::real>* mElementalSMatrix;
    Matrix<dtype::real>* mElementalRMatrix;
    dtype::size mNumHarmonics;
};

#endif
