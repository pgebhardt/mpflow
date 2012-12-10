// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <vector>
#include <array>
#include <tuple>
#include <memory>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../include/dtype.h"
#include "../include/math.h"
#include "../include/matrix.h"
#include "../include/sparse_matrix.h"
#include "../include/basis.h"
#include "../include/mesh.h"
#include "../include/electrodes.h"
#include "../include/model.h"
#include "../include/model_cuda.h"

// create solver model
template <
    class BasisFunction
>
fastEIT::Model<BasisFunction>::Model(std::shared_ptr<Mesh<BasisFunction>> mesh, std::shared_ptr<Electrodes> electrodes,
    dtype::real sigmaRef, dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream)
    : mesh_(mesh), electrodes_(electrodes), sigma_ref_(sigmaRef), s_matrix_(nullptr), r_matrix_(nullptr),
        excitation_matrix_(nullptr), connectivity_matrix_(nullptr), elemental_s_matrix_(nullptr),
        elemental_r_matrix_(nullptr), num_harmonics_(numHarmonics) {
    // check input
    if (mesh == NULL) {
        throw std::invalid_argument("Model::Model: mesh == NULL");
    }
    if (electrodes == NULL) {
        throw std::invalid_argument("Model::Model: electrodes == NULL");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Model::Model: handle == NULL");
    }

    // create sparse matrices
    this->createSparseMatrices(handle, stream);

    // create matrices
    this->excitation_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        this->electrodes()->count(), stream);
    this->connectivity_matrix_ = std::make_shared<Matrix<dtype::index>>(this->mesh()->nodes()->rows(),
        SparseMatrix::block_size * Matrix<dtype::real>::block_size, stream);
    this->elemental_s_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        SparseMatrix::block_size * Matrix<dtype::real>::block_size, stream);
    this->elemental_r_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        SparseMatrix::block_size * Matrix<dtype::real>::block_size, stream);

    // init model
    this->init(handle, stream);

    // init excitaion matrix
    this->initExcitationMatrix(stream);
}

// create sparse matrices
template <
    class BasisFunction
>
void fastEIT::Model<BasisFunction>::createSparseMatrices(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::createSparseMatrices: handle == NULL");
    }

    // calc initial system matrix
    // create matrices
    Matrix<dtype::real> system_matrix(this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);

    // calc generate empty system matrix
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get nodes for element
        indices = this->mesh()->elementIndices(element);

        // set system matrix elements
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; ++i) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; ++j) {
                system_matrix(indices[i], indices[j]) = 1.0f;
            }
        }
    }

    // copy matrix to device
    system_matrix.copyToDevice(stream);

    // create sparse matrices
    this->s_matrix_ = std::make_shared<fastEIT::SparseMatrix>(system_matrix, stream);
    this->r_matrix_ = std::make_shared<fastEIT::SparseMatrix>(system_matrix, stream);

    for (dtype::index harmonic = 0; harmonic < this->num_harmonics() + 1; ++harmonic) {
        this->system_matrices_.push_back(std::make_shared<fastEIT::SparseMatrix>(system_matrix, stream));
    }
}

// init model
template <
    class BasisFunction
>
void fastEIT::Model<BasisFunction>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::init: handle == NULL");
    }

    // create intermediate matrices
    Matrix<dtype::index> element_count(this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    Matrix<dtype::index> connectivity_matrix(this->connectivity_matrix()->data_rows(),
        element_count.data_columns() * fastEIT::Matrix<dtype::index>::block_size, stream);
    Matrix<dtype::real> elemental_s_matrix(this->elemental_s_matrix()->data_rows(),
        element_count.data_columns() * fastEIT::Matrix<dtype::real>::block_size, stream);
    Matrix<dtype::real> elemental_r_matrix(this->elemental_r_matrix()->data_rows(),
        element_count.data_columns() * fastEIT::Matrix<dtype::real>::block_size, stream);

    // init connectivityMatrix
    for (dtype::size i = 0; i < connectivity_matrix.rows(); ++i) {
        for (dtype::size j = 0; j < connectivity_matrix.columns(); ++j) {
            connectivity_matrix(i, j) = -1;
        }
    }
    for (dtype::size i = 0; i < this->connectivity_matrix()->rows(); ++i) {
        for (dtype::size j = 0; j < this->connectivity_matrix()->columns(); ++j) {
            (*this->connectivity_matrix())(i, j) =  -1;
        }
    }
    this->connectivity_matrix()->copyToDevice(stream);

    // fill intermediate connectivity and elemental matrices
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;
    std::array<BasisFunction*, BasisFunction::nodes_per_element> basis_functions;
    dtype::real temp;

    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get element indices
        indices = this->mesh()->elementIndices(element);

        // calc corresponding basis functions
        for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
            basis_functions[node] = new BasisFunction(this->mesh()->elementNodes(element), node);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; i++) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; j++) {
                // get current element count
                temp = element_count(indices[i], indices[j]);

                // set connectivity element
                connectivity_matrix(indices[i], indices[j] + connectivity_matrix.data_rows() * temp) = element;

                // set elemental system element
                elemental_s_matrix(indices[i], indices[j] + connectivity_matrix.data_rows() * temp) =
                    basis_functions[i]->integrateGradientWithBasis(*basis_functions[j]);

                // set elemental residual element
                elemental_r_matrix(indices[i], indices[j] + connectivity_matrix.data_rows() * temp) =
                    basis_functions[i]->integrateWithBasis(*basis_functions[j]);

                // increment element count
                element_count(indices[i], indices[j])++;
            }
        }

        // cleanup
        for (BasisFunction*& basis : basis_functions) {
            delete basis;
        }
    }

    // upload intermediate matrices
    connectivity_matrix.copyToDevice(stream);
    elemental_s_matrix.copyToDevice(stream);
    elemental_r_matrix.copyToDevice(stream);

    // reduce matrices
    model::reduceMatrix(&connectivity_matrix, this->s_matrix().get(), stream,
        this->connectivity_matrix().get());
    model::reduceMatrix(&elemental_s_matrix, this->s_matrix().get(), stream,
        this->elemental_s_matrix().get());
    model::reduceMatrix(&elemental_r_matrix, this->s_matrix().get(), stream,
        this->elemental_r_matrix().get());

    // create gamma
    Matrix<dtype::real> gamma(this->mesh()->elements()->rows(), 1, stream);

    // update matrices
    model::updateMatrix(this->elemental_s_matrix().get(), &gamma, this->connectivity_matrix().get(),
        this->sigma_ref(), stream, this->s_matrix().get());
    model::updateMatrix(this->elemental_r_matrix().get(), &gamma, this->connectivity_matrix().get(),
        this->sigma_ref(), stream, this->r_matrix().get());
}

// update model
template <
    class BasisFunction
>
void fastEIT::Model<BasisFunction>::update(const std::shared_ptr<Matrix<dtype::real>> gamma, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::init: handle == NULL");
    }

    // update matrices
    model::updateMatrix(this->elemental_s_matrix().get(), gamma.get(), this->connectivity_matrix().get(),
        this->sigma_ref(), stream, this->s_matrix().get());
    model::updateMatrix(this->elemental_r_matrix().get(), gamma.get(), this->connectivity_matrix().get(),
        this->sigma_ref(), stream, this->r_matrix().get());

    // set cublas stream
    cublasSetStream(handle, stream);

    // create system matrices for all harmonics
    dtype::real alpha = 0.0f;
    for (dtype::index harmonic = 0; harmonic < this->num_harmonics() + 1; ++harmonic) {
        // calc alpha
        alpha = fastEIT::math::square(2.0f * harmonic * M_PI / this->mesh()->height());

        // init system matrix with 2d system matrix
        if (cublasScopy(handle, this->s_matrix()->data_rows() * SparseMatrix::block_size,
            this->s_matrix()->values(), 1, this->system_matrix(harmonic)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "Model::update: calc system matrices for all harmonics");
        }

        // add alpha * residualMatrix
        if (cublasSaxpy(handle, this->s_matrix()->data_rows() * SparseMatrix::block_size, &alpha,
            this->r_matrix()->values(), 1, this->system_matrix(harmonic)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "Model::update: calc system matrices for all harmonics");
        }
    }
}

// init exitation matrix
template <
    class BasisFunction
>
void fastEIT::Model<BasisFunction>::initExcitationMatrix(cudaStream_t stream) {
    // fill exitation_matrix matrix
    std::array<dtype::index, BasisFunction::nodes_per_edge> indices;
    std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_edge> nodes,
        permutated_nodes;

    for (dtype::index boundary_element = 0;
        boundary_element < this->mesh()->boundary()->rows();
        ++boundary_element) {
        for (dtype::index electrode = 0;
            electrode < this->electrodes()->count();
            ++electrode) {
            // get indices
            indices = this->mesh()->boundaryIndices(boundary_element);

            // get nodes
            nodes = this->mesh()->boundaryNodes(boundary_element);

            // calc elements
            for (dtype::index node = 0; node < BasisFunction::nodes_per_edge; ++node) {
                // permutate nodes
                for (dtype::index n = 0; n < BasisFunction::nodes_per_edge; ++n) {
                    permutated_nodes[n] = nodes[(node + n) % BasisFunction::nodes_per_edge];
                }

                // add new value
                (*this->excitation_matrix())(indices[node], electrode) -= BasisFunction::integrateBoundaryEdge(
                    permutated_nodes, this->electrodes()->electrodes_start()[electrode],
                    this->electrodes()->electrodes_end()[electrode]) / this->electrodes()->width();
            }
        }
    }

    // upload matrix
    this->excitation_matrix()->copyToDevice(stream);
}

// calc excitaion components
template <
    class BasisFunction
>
void fastEIT::Model<BasisFunction>::calcExcitationComponent(const std::shared_ptr<Matrix<dtype::real>> pattern,
    dtype::size harmonic, cublasHandle_t handle, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> component) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::calcExcitationComponents: handle == NULL");
    }
    if (component == NULL) {
        throw std::invalid_argument("Model::calcExcitationComponents: components == NULL");
    }

    // calc excitation matrices
    // Run multiply once more to avoid cublas error
    try {
        component->multiply(*this->excitation_matrix(), *pattern, handle, stream);
    }
    catch(const std::exception& e) {
        component->multiply(*this->excitation_matrix(), *pattern, handle, stream);
    }

    // calc fourier coefficients for current pattern
    if (harmonic == 0) {
        // calc ground mode
        component->scalarMultiply(1.0f / this->mesh()->height(), stream);
    } else {
        component->scalarMultiply(
            2.0f * sin(harmonic * M_PI * this->electrodes()->height() / this->mesh()->height()) /
            (harmonic * M_PI * this->electrodes()->height()), stream);
    }
}

// specialisation
template class fastEIT::Model<fastEIT::basis::Linear>;
