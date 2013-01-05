// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"
#include "../include/model_kernel.h"

// create solver model
template <
    class BasisFunction
>
fastEIT::Model<BasisFunction>::Model(std::shared_ptr<Mesh<BasisFunction>> mesh, std::shared_ptr<Electrodes> electrodes,
    dtype::real sigmaRef, dtype::size numHarmonics, cublasHandle_t handle, cudaStream_t stream)
    : mesh_(mesh), electrodes_(electrodes), sigma_ref_(sigmaRef), num_harmonics_(numHarmonics) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("Model::Model: mesh == nullptr");
    }
    if (electrodes == nullptr) {
        throw std::invalid_argument("Model::Model: electrodes == nullptr");
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
        sparseMatrix::block_size * matrix::block_size, stream);
    this->elemental_s_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * matrix::block_size, stream);
    this->elemental_r_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * matrix::block_size, stream);

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
    auto system_matrix = std::make_shared<Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);

    // calc generate empty system matrix
    std::array<dtype::index, BasisFunction::nodes_per_element> indices;
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get nodes for element
        indices = this->mesh()->elementIndices(element);

        // set system matrix elements
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; ++i) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; ++j) {
                (*system_matrix)(indices[i], indices[j]) = 1.0f;
            }
        }
    }

    // copy matrix to device
    system_matrix->copyToDevice(stream);

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
    auto element_count = std::make_shared<Matrix<dtype::index>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    auto connectivity_matrix = std::make_shared<Matrix<dtype::index>>(
        this->connectivity_matrix()->data_rows(),
        element_count->data_columns() * fastEIT::matrix::block_size, stream);
    auto elemental_s_matrix = std::make_shared<Matrix<dtype::real>>(
        this->elemental_s_matrix()->data_rows(),
        element_count->data_columns() * fastEIT::matrix::block_size, stream);
    auto elemental_r_matrix = std::make_shared<Matrix<dtype::real>>(
        this->elemental_r_matrix()->data_rows(),
        element_count->data_columns() * fastEIT::matrix::block_size, stream);

    // init connectivityMatrix
    for (dtype::size i = 0; i < connectivity_matrix->rows(); ++i) {
        for (dtype::size j = 0; j < connectivity_matrix->columns(); ++j) {
            (*connectivity_matrix)(i, j) = -1;
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
    std::array<std::shared_ptr<BasisFunction>, BasisFunction::nodes_per_element> basis_functions;
    dtype::real temp;

    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get element indices
        indices = this->mesh()->elementIndices(element);

        // calc corresponding basis functions
        for (dtype::index node = 0; node < BasisFunction::nodes_per_element; ++node) {
            basis_functions[node] = std::make_shared<BasisFunction>(
                this->mesh()->elementNodes(element), node);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < BasisFunction::nodes_per_element; i++) {
            for (dtype::index j = 0; j < BasisFunction::nodes_per_element; j++) {
                // get current element count
                temp = (*element_count)(indices[i], indices[j]);

                // set connectivity element
                (*connectivity_matrix)(indices[i], indices[j] + connectivity_matrix->data_rows() * temp) = element;

                // set elemental system element
                (*elemental_s_matrix)(indices[i], indices[j] + connectivity_matrix->data_rows() * temp) =
                    basis_functions[i]->integrateGradientWithBasis(basis_functions[j]);

                // set elemental residual element
                (*elemental_r_matrix)(indices[i], indices[j] + connectivity_matrix->data_rows() * temp) =
                    basis_functions[i]->integrateWithBasis(basis_functions[j]);

                // increment element count
                (*element_count)(indices[i], indices[j])++;
            }
        }
    }

    // upload intermediate matrices
    connectivity_matrix->copyToDevice(stream);
    elemental_s_matrix->copyToDevice(stream);
    elemental_r_matrix->copyToDevice(stream);

    // reduce matrices
    model::reduceMatrix(connectivity_matrix, this->s_matrix(), stream,
        this->connectivity_matrix());
    model::reduceMatrix(elemental_s_matrix, this->s_matrix(), stream,
        this->elemental_s_matrix());
    model::reduceMatrix(elemental_r_matrix, this->s_matrix(), stream,
        this->elemental_r_matrix());

    // create gamma
    auto gamma = std::make_shared<Matrix<dtype::real>>(this->mesh()->elements()->rows(), 1, stream);

    // update matrices
    model::updateMatrix(this->elemental_s_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->s_matrix());
    model::updateMatrix(this->elemental_r_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->r_matrix());
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
    model::updateMatrix(this->elemental_s_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->s_matrix());
    model::updateMatrix(this->elemental_r_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->r_matrix());

    // set cublas stream
    cublasSetStream(handle, stream);

    // create system matrices for all harmonics
    dtype::real alpha = 0.0f;
    for (dtype::index harmonic = 0; harmonic < this->num_harmonics() + 1; ++harmonic) {
        // calc alpha
        alpha = fastEIT::math::square(2.0f * harmonic * M_PI / this->mesh()->height());

        // init system matrix with 2d system matrix
        if (cublasScopy(handle, this->s_matrix()->data_rows() * sparseMatrix::block_size,
            this->s_matrix()->values(), 1, this->system_matrix(harmonic)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "Model::update: calc system matrices for all harmonics");
        }

        // add alpha * residualMatrix
        if (cublasSaxpy(handle, this->s_matrix()->data_rows() * sparseMatrix::block_size, &alpha,
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
    std::array<dtype::index, BasisFunction::nodes_per_edge> nodes_indices;
    std::array<std::tuple<dtype::real, dtype::real>, BasisFunction::nodes_per_edge> nodes_coordinates;

    // needed arrays
    std::array<dtype::real, BasisFunction::nodes_per_edge> node_parameter;
    std::array<dtype::real, 2> boundary_parameter;

    // calc excitaion matrix
    for (dtype::index boundary_element = 0;
        boundary_element < this->mesh()->boundary()->rows();
        ++boundary_element) {
        // get indices
        nodes_indices = this->mesh()->boundaryIndices(boundary_element);

        // get nodes coordinates
        nodes_coordinates = this->mesh()->boundaryNodes(boundary_element);

        // prepare vector to sort nodes by parameter
        std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
        for (dtype::size node = 0; node < BasisFunction::nodes_per_edge; ++node) {
            nodes.push_back(std::make_tuple(nodes_indices[node], nodes_coordinates[node]));
        }

        // sort nodes by parameter
        std::sort(nodes.begin(), nodes.end(),
            [](const std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>& a,
                const std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>& b)
                -> bool {
                    return math::circleParameter(std::get<1>(b),
                        math::circleParameter(std::get<1>(a), 0.0)) > 0.0;
        });

        // calc parameter offset
        dtype::real parameter_offset = math::circleParameter(std::get<1>(nodes[0]), 0.0);

        // calc node parameter centered to node
        for (dtype::size i = 0; i < BasisFunction::nodes_per_edge; ++i) {
            node_parameter[i] = math::circleParameter(std::get<1>(nodes[i]),
                parameter_offset);
        }

        // integrate basis function for each electrode
        for (dtype::index electrode = 0;
            electrode < this->electrodes()->count();
            ++electrode) {
            // calc boundary parameter centered to node
            boundary_parameter[0] = math::circleParameter(this->electrodes()->electrodes_start()[electrode],
                parameter_offset);
            boundary_parameter[1] = math::circleParameter(this->electrodes()->electrodes_end()[electrode],
                parameter_offset);

            // calc elements
            for (dtype::index node = 0; node < BasisFunction::nodes_per_edge; ++node) {
                (*this->excitation_matrix())(std::get<0>(nodes[node]), electrode) -=
                    BasisFunction::integrateBoundaryEdge(
                        node_parameter, node, boundary_parameter[0], boundary_parameter[1]) /
                    this->electrodes()->width();
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
    if (component == nullptr) {
        throw std::invalid_argument("Model::calcExcitationComponents: components == nullptr");
    }

    // calc excitation matrices
    // Run multiply once more to avoid cublas error
    try {
        component->multiply(this->excitation_matrix(), pattern, handle, stream);
    }
    catch(const std::exception& e) {
        component->multiply(this->excitation_matrix(), pattern, handle, stream);
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

// reduce matrix
template <
    class type
>
void fastEIT::model::reduceMatrix(const std::shared_ptr<Matrix<type>> intermediateMatrix,
    const std::shared_ptr<SparseMatrix> shape, cudaStream_t stream,
    std::shared_ptr<Matrix<type>> matrix) {
    // check input
    if (intermediateMatrix == nullptr) {
        throw std::invalid_argument("model::reduceMatrix: intermediateMatrix == nullptr");
    }
    if (shape == nullptr) {
        throw std::invalid_argument("model::reduceMatrix: shape == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("model::reduceMatrix: matrix == nullptr");
    }

    // block size
    dim3 blocks(matrix->data_rows() / matrix::block_size, 1);
    dim3 threads(matrix::block_size, matrix::block_size);

    // reduce matrix
    modelKernel::reduceMatrix<type>(blocks, threads, stream,
        intermediateMatrix->device_data(), shape->column_ids(), matrix->data_rows(),
        shape->density(), matrix->device_data());
}

// update matrix
void fastEIT::model::updateMatrix(const std::shared_ptr<Matrix<dtype::real>> elements,
    const std::shared_ptr<Matrix<dtype::real>> gamma,
    const std::shared_ptr<Matrix<dtype::index>> connectivityMatrix,
    dtype::real sigmaRef, cudaStream_t stream,
    std::shared_ptr<SparseMatrix> matrix) {
    // check input
    if (elements == nullptr) {
        throw std::invalid_argument("model::updateMatrix: elements == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("model::updateMatrix: gamma == nullptr");
    }
    if (connectivityMatrix == nullptr) {
        throw std::invalid_argument("model::updateMatrix: connectivityMatrix == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("model::updateMatrix: matrix == nullptr");
    }

    // dimension
    dim3 threads(matrix::block_size, matrix::block_size);
    dim3 blocks(matrix->data_rows() / matrix::block_size, 1);

    // execute kernel
    modelKernel::updateMatrix(blocks, threads, stream,
        connectivityMatrix->device_data(), elements->device_data(), gamma->device_data(),
        sigmaRef, connectivityMatrix->data_rows(), matrix->density(), matrix->values());
}

// specialisation
template void fastEIT::model::reduceMatrix<fastEIT::dtype::real>(const std::shared_ptr<Matrix<fastEIT::dtype::real>>,
    const std::shared_ptr<SparseMatrix>, cudaStream_t, std::shared_ptr<Matrix<fastEIT::dtype::real>>);
template void fastEIT::model::reduceMatrix<fastEIT::dtype::index>(const std::shared_ptr<Matrix<fastEIT::dtype::index>>,
    const std::shared_ptr<SparseMatrix>, cudaStream_t, std::shared_ptr<Matrix<fastEIT::dtype::index>>);

template class fastEIT::Model<fastEIT::basis::Linear>;
