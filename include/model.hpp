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
        dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream);
    virtual ~Model();

// init methods
private:
    void init(cublasHandle_t handle, cudaStream_t stream);
    void create_sparse_matrices(cublasHandle_t handle, cudaStream_t stream);
    void init_excitation_matrix(cudaStream_t stream);

public:
    // calc excitaion components
    void calc_excitation_components(linalgcuMatrix_t* component, linalgcuMatrix_t pattern,
        cublasHandle_t handle, cudaStream_t stream);

    // update model
    void update(linalgcuMatrix_t gamma, cublasHandle_t handle, cudaStream_t stream);

// cuda methods
private:
    // update matrix
    void update_matrix(linalgcuSparseMatrix_t matrix, linalgcuMatrix_t elements,
        linalgcuMatrix_t gamma, cudaStream_t stream);

    // reduce matrix
    void reduce_matrix(linalgcuMatrix_t matrix, linalgcuMatrix_t intermediateMatrix,
        dtype::size density, cudaStream_t stream);

// access methods
public:
    Mesh* mesh() const { return this->mMesh; }
    Electrodes* electrodes() const { return this->mElectrodes; }
    dtype::real sigmaRef() const { return this->mSigmaRef; }
    inline linalgcuSparseMatrix_t systemMatrix(dtype::size id) {
        assert(id <= this->mNumHarmonics);
        return this->mSystemMatrix[id];
    }
    linalgcuMatrix_t excitationMatrix() { return this->mExcitationMatrix; }
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
    linalgcuSparseMatrix_t* mSystemMatrix;
    linalgcuSparseMatrix_t mSMatrix;
    linalgcuSparseMatrix_t mRMatrix;
    linalgcuMatrix_t mExcitationMatrix;
    linalgcuMatrix_t mConnectivityMatrix;
    linalgcuMatrix_t mElementalSMatrix;
    linalgcuMatrix_t mElementalRMatrix;
    dtype::size mNumHarmonics;
};

#endif
