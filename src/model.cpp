// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <vector>
#include <array>
#include <tuple>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.hpp"
#include "../include/math.hpp"
#include "../include/matrix.hpp"
#include "../include/sparse.hpp"
#include "../include/basis.hpp"
#include "../include/mesh.hpp"
#include "../include/electrodes.hpp"
#include "../include/model.hpp"
#include "../include/model.hcu"

// create solver model
template <class BasisFunction>
fastEIT::Model<BasisFunction>::Model(Mesh<BasisFunction>& mesh, Electrodes& electrodes, dtype::real sigmaRef,
    dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream)
    : mesh_(&mesh), electrodes_(&electrodes), sigma_ref_(sigmaRef), s_matrix_(NULL), r_matrix_(NULL),
        excitation_matrix_(NULL), connectivity_matrix_(NULL), elemental_s_matrix_(NULL),
        elemental_r_matrix_(NULL), num_harmonics_(numHarmonics) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::Model: handle == NULL");
    }

    // create sparse matrices
    this->createSparseMatrices(handle, stream);

    // create matrices
    this->excitation_matrix_ = new Matrix<dtype::real>(this->mesh().nodes().rows(),
        this->electrodes().count(), stream);
    this->connectivity_matrix_ = new Matrix<dtype::index>(this->mesh().nodes().rows(),
        SparseMatrix::block_size * Matrix<dtype::real>::block_size, stream);
    this->elemental_s_matrix_ = new Matrix<dtype::real>(this->mesh().nodes().rows(),
        SparseMatrix::block_size * Matrix<dtype::real>::block_size, stream);
    this->elemental_r_matrix_ = new Matrix<dtype::real>(this->mesh().nodes().rows(),
        SparseMatrix::block_size * Matrix<dtype::real>::block_size, stream);

    // init model
    this->init(handle, stream);

    // init excitaion matrix
    this->initExcitationMatrix(stream);
}

// release solver model
template <class BasisFunction>
fastEIT::Model<BasisFunction>::~Model() {
    // cleanup
    delete this->mesh_;
    delete this->electrodes_;
    delete this->s_matrix_;
    delete this->r_matrix_;
    delete this->excitation_matrix_;
    delete this->connectivity_matrix_;
    delete this->elemental_s_matrix_;
    delete this->elemental_r_matrix_;

    for (dtype::index i = 0; i < this->num_harmonics() + 1; ++i) {
        delete this->set_system_matrices()[i];
    }
}


// create sparse matrices
template <class BasisFunction>
void fastEIT::Model<BasisFunction>::createSparseMatrices(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::createSparseMatrices: handle == NULL");
    }

    // calc initial system matrix
    // create matrices
    Matrix<dtype::real> systemMatrix(this->mesh().nodes().rows(), this->mesh().nodes().rows(), stream);

    // calc generate empty system matrix
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;
    for (dtype::index element = 0; element < this->mesh().elements().rows(); ++element) {
        // get nodes for element
        indices = this->mesh().elementIndices(element);

        // set system matrix elements
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; ++i) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; ++j) {
                systemMatrix.set(indices[i], indices[j]) = 1.0f;
            }
        }
    }

    // copy matrix to device
    systemMatrix.copyToDevice(stream);

    // create sparse matrices
    this->s_matrix_ = new fastEIT::SparseMatrix(systemMatrix, stream);
    this->r_matrix_ = new fastEIT::SparseMatrix(systemMatrix, stream);

    for (dtype::index harmonic = 0; harmonic < this->num_harmonics() + 1; ++harmonic) {
        this->set_system_matrices().push_back(new fastEIT::SparseMatrix(systemMatrix, stream));
    }
}

// init model
template <class BasisFunction>
void fastEIT::Model<BasisFunction>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::init: handle == NULL");
    }

    // create intermediate matrices
    Matrix<dtype::index> elementCount(this->mesh().nodes().rows(), this->mesh().nodes().rows(), stream);
    Matrix<dtype::index> connectivityMatrix(this->connectivity_matrix().data_rows(),
        elementCount.data_columns() * fastEIT::Matrix<dtype::index>::block_size, stream);
    Matrix<dtype::real> elementalSMatrix(this->elemental_s_matrix().data_rows(),
        elementCount.data_columns() * fastEIT::Matrix<dtype::real>::block_size, stream);
    Matrix<dtype::real> elementalRMatrix(this->elemental_r_matrix().data_rows(),
        elementCount.data_columns() * fastEIT::Matrix<dtype::real>::block_size, stream);

    // init connectivityMatrix
    for (dtype::size i = 0; i < connectivityMatrix.rows(); ++i) {
        for (dtype::size j = 0; j < connectivityMatrix.columns(); ++j) {
            connectivityMatrix.set(i, j) = -1;
        }
    }
    for (dtype::size i = 0; i < this->connectivity_matrix().rows(); ++i) {
        for (dtype::size j = 0; j < this->connectivity_matrix().columns(); ++j) {
            this->set_connectivity_matrix().set(i, j) =  -1;
        }
    }
    this->set_connectivity_matrix().copyToDevice(stream);

    // fill intermediate connectivity and elemental matrices
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;
    std::array<BasisFunction*, BasisFunction::nodes_per_element> basis_functions;
    dtype::real temp;

    for (dtype::index element = 0; element < this->mesh().elements().rows(); ++element) {
        // get element indices
        indices = this->mesh().elementIndices(element);

        // calc corresponding basis functions
        for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
            basis_functions[node] = new BasisFunction(this->mesh().elementNodes(element), node);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; i++) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; j++) {
                // get current element count
                temp = elementCount.get(indices[i], indices[j]);

                // set connectivity element
                connectivityMatrix.set(indices[i], indices[j] + connectivityMatrix.data_rows() * temp) = element;

                // set elemental system element
                elementalSMatrix.set(indices[i], indices[j] + connectivityMatrix.data_rows() * temp) =
                    basis_functions[i]->integrateGradientWithBasis(*basis_functions[j]);

                // set elemental residual element
                elementalRMatrix.set(indices[i], indices[j] + connectivityMatrix.data_rows() * temp) =
                    basis_functions[i]->integrateWithBasis(*basis_functions[j]);

                // increment element count
                elementCount.set(indices[i], indices[j])++;
            }
        }

        // cleanup
        for (BasisFunction*& basis : basis_functions) {
            delete basis;
        }
    }

    // upload intermediate matrices
    connectivityMatrix.copyToDevice(stream);
    elementalSMatrix.copyToDevice(stream);
    elementalRMatrix.copyToDevice(stream);

    // reduce matrices
    model::reduceMatrix(connectivityMatrix, this->s_matrix(), stream,
        this->set_connectivity_matrix());
    model::reduceMatrix(elementalSMatrix, this->s_matrix(), stream,
        this->set_elemental_s_matrix());
    model::reduceMatrix(elementalRMatrix, this->s_matrix(), stream,
        this->set_elemental_r_matrix());

    // create gamma
    Matrix<dtype::real> gamma(this->mesh().elements().rows(), 1, stream);

    // update matrices
    model::updateMatrix(this->elemental_s_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->set_s_matrix());
    model::updateMatrix(this->elemental_r_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->set_r_matrix());
}

// update model
template <class BasisFunction>
void fastEIT::Model<BasisFunction>::update(const Matrix<dtype::real>& gamma, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::init: handle == NULL");
    }

    // update matrices
    model::updateMatrix(this->elemental_s_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->set_s_matrix());
    model::updateMatrix(this->elemental_r_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->set_r_matrix());

    // set cublas stream
    cublasSetStream(handle, stream);

    // create system matrices for all harmonics
    dtype::real alpha = 0.0f;
    for (dtype::index harmonic = 0; harmonic < this->num_harmonics() + 1; ++harmonic) {
        // calc alpha
        alpha = fastEIT::math::square(2.0f * harmonic * M_PI / this->mesh().height());

        // init system matrix with 2d system matrix
        if (cublasScopy(handle, this->s_matrix().data_rows() * SparseMatrix::block_size,
            this->set_s_matrix().values(), 1, this->system_matrices()[harmonic]->set_values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "Model::update: calc system matrices for all harmonics");
        }

        // add alpha * residualMatrix
        if (cublasSaxpy(handle, this->s_matrix().data_rows() * SparseMatrix::block_size, &alpha,
            this->set_r_matrix().values(), 1, this->system_matrices()[harmonic]->set_values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "Model::update: calc system matrices for all harmonics");
        }
    }
}

// init exitation matrix
template <class BasisFunction>
void fastEIT::Model<BasisFunction>::initExcitationMatrix(cudaStream_t stream) {
    // fill exitation_matrix matrix
    std::array<dtype::index, BasisFunction::nodes_per_edge> indices;
    std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_edge> nodes,
        permutated_nodes;

    for (dtype::index i = 0; i < this->mesh().boundary().rows(); ++i) {
        for (dtype::index l = 0; l < this->electrodes().count(); ++l) {
            // get indices
            indices = this->mesh().boundaryIndices(i);

            // get nodes
            nodes = this->mesh().boundaryNodes(i);

            // calc elements
            for (dtype::index k = 0; k < BasisFunction::nodes_per_edge; ++k) {
                // permutate nodes
                for (dtype::index n = 0; n < BasisFunction::nodes_per_edge; ++n) {
                    permutated_nodes[n] = nodes[(k + n) % BasisFunction::nodes_per_edge];
                }

                // add new value
                this->set_excitation_matrix().set(indices[k], l) -= BasisFunction::integrateBoundaryEdge(permutated_nodes,
                    this->electrodes().electrodes_start()[l], this->electrodes().electrodes_end()[l]) /
                    this->electrodes().width();
            }
        }
    }

    // upload matrix
    this->set_excitation_matrix().copyToDevice(stream);
}

// calc excitaion components
template <class BasisFunction>
void fastEIT::Model<BasisFunction>::calcExcitationComponents(const Matrix<dtype::real>& pattern,
    cublasHandle_t handle, cudaStream_t stream, std::vector<Matrix<dtype::real>*>& components) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::calcExcitationComponents: handle == NULL");
    }

    // calc excitation matrices
    for (dtype::index harmonic = 0; harmonic < this->num_harmonics() + 1; ++harmonic) {
        // Run multiply once more to avoid cublas error
        try {
            components[harmonic]->multiply(this->excitation_matrix(), pattern, handle, stream);
        }
        catch (std::exception& e) {
            components[harmonic]->multiply(this->excitation_matrix(), pattern, handle, stream);
        }
    }

    // calc fourier coefficients for current pattern
    // calc ground mode
    components[0]->scalarMultiply(1.0f / this->mesh().height(), stream);

    // calc harmonics
    for (dtype::index harmonic = 1; harmonic < this->num_harmonics() + 1; ++harmonic) {
        components[harmonic]->scalarMultiply(
            2.0f * sin(harmonic * M_PI * this->electrodes().height() / this->mesh().height()) /
            (harmonic * M_PI * this->electrodes().height()), stream);
    }
}

// specialisation
template class fastEIT::Model<fastEIT::basis::Linear>;
