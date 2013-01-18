// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"
#include "../include/model_kernel.h"

// create solver model
template <
    class basis_function_type
>
fastEIT::Model<basis_function_type>::Model(
    std::shared_ptr<Mesh<basis_function_type>> mesh,
    std::shared_ptr<Electrodes<Mesh<basis_function_type>>> electrodes,
    std::shared_ptr<source::Source> source, dtype::real sigmaRef,
    dtype::size components_count, cublasHandle_t handle, cudaStream_t stream)
    : mesh_(mesh), electrodes_(electrodes), source_(source), sigma_ref_(sigmaRef),
        components_count_(components_count) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("Model::Model: mesh == nullptr");
    }
    if (electrodes == nullptr) {
        throw std::invalid_argument("Model::Model: electrodes == nullptr");
    }
    if (source == nullptr) {
        throw std::invalid_argument("Model::Model: source == nullptr");
    }
    if (components_count == 0) {
        throw std::invalid_argument("Model::Model: components_count == 0");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Model::Model: handle == NULL");
    }

    // create matrices
    this->connectivity_matrix_ = std::make_shared<Matrix<dtype::index>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * matrix::block_size, stream);
    this->elemental_s_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * matrix::block_size, stream);
    this->elemental_r_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * matrix::block_size, stream);
    this->z_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        this->mesh()->nodes()->rows(), stream);

    for (dtype::index component = 0; component < this->components_count() + 1; ++component) {
        this->potential_.push_back(std::make_shared<Matrix<dtype::real>>(
            this->mesh()->nodes()->rows() + this->electrodes()->count(),
            this->source()->drive_count() + this->source()->measurement_count(), stream));
    }

    // init model
    this->init(handle, stream);
}

// init model
template <
    class basis_function_type
>
void fastEIT::Model<basis_function_type>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("Model::init: handle == NULL");
    }

    // create gamma
    auto gamma = std::make_shared<Matrix<dtype::real>>(this->mesh()->elements()->rows(), 1, stream);

    // init elemental matrices
    auto common_element_matrix = this->initElementalMatrices(stream);


    // init complete electrode model
    std::shared_ptr<Matrix<dtype::real>> w_matrix, d_matrix;
    std::tie(w_matrix, d_matrix) = this->initCEMMatrices(stream);

    // update r and s matrices
    model::updateMatrix(this->elemental_s_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->s_matrix());
    model::updateMatrix(this->elemental_r_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->r_matrix());

    // assamble initial system matrices
    auto system_matrix = std::make_shared<Matrix<dtype::real>>(
        this->mesh()->nodes()->rows() + this->electrodes()->count(),
        this->mesh()->nodes()->rows() + this->electrodes()->count(),
        stream);

    // fill s + r + z
    for (dtype::index row = 0; row < this->mesh()->nodes()->rows(); ++row) {
        for (dtype::index column = 0; column < this->mesh()->nodes()->rows(); ++column) {
            (*system_matrix)(row, column) = (*common_element_matrix)(row, column);
        }
    }

    // fill w and wT
    for (dtype::index node = 0; node < this->mesh()->nodes()->rows(); ++node) {
        for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
            (*system_matrix)(node, electrode + this->mesh()->nodes()->rows()) =
                (*w_matrix)(node, electrode);

            (*system_matrix)(electrode + this->mesh()->nodes()->rows(), node) =
                (*w_matrix)(node, electrode);
        }
    }

    // fill d matrix
    for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
        (*system_matrix)(electrode + this->mesh()->nodes()->rows(), electrode + this->mesh()->nodes()->rows()) =
            (*d_matrix)(electrode, electrode);
    }

    // create sparse matrices
    system_matrix->copyToDevice(stream);
    for (dtype::index component = 0; component < this->components_count(); ++component) {
        this->system_matrices_.push_back(std::make_shared<SparseMatrix>(system_matrix, stream));
    }
}

// init elemental matrices
template <
    class basis_function_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::Model<basis_function_type>::initElementalMatrices(
    cudaStream_t stream) {
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
    std::array<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>, basis_function_type::nodes_per_element> nodes;
    std::array<std::tuple<dtype::real, dtype::real>, basis_function_type::nodes_per_element> nodes_coordinates;
    std::array<std::shared_ptr<basis_function_type>, basis_function_type::nodes_per_element> basis_functions;
    dtype::real temp;

    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get element nodes
        nodes = this->mesh()->elementNodes(element);

        // extract coordinates
        for (dtype::index node = 0; node < basis_function_type::nodes_per_element; ++node) {
            nodes_coordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < basis_function_type::nodes_per_element; ++node) {
            basis_functions[node] = std::make_shared<basis_function_type>(
                nodes_coordinates, node);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; i++) {
            for (dtype::index j = 0; j < basis_function_type::nodes_per_element; j++) {
                // get current element count
                temp = (*element_count)(std::get<0>(nodes[i]), std::get<0>(nodes[j]));

                // set connectivity element
                (*connectivity_matrix)(std::get<0>(nodes[i]), std::get<0>(
                    nodes[j]) + connectivity_matrix->data_rows() * temp) = element;

                // set elemental system element
                (*elemental_s_matrix)(std::get<0>(nodes[i]), std::get<0>(
                    nodes[j]) + connectivity_matrix->data_rows() * temp) =
                    basis_functions[i]->integrateGradientWithBasis(basis_functions[j]);

                // set elemental residual element
                (*elemental_r_matrix)(std::get<0>(nodes[i]), std::get<0>(
                    nodes[j]) + connectivity_matrix->data_rows() * temp) =
                    basis_functions[i]->integrateWithBasis(basis_functions[j]);

                // increment element count
                (*element_count)(std::get<0>(nodes[i]), std::get<0>(nodes[j]))++;
            }
        }
    }

    // upload intermediate matrices
    connectivity_matrix->copyToDevice(stream);
    elemental_s_matrix->copyToDevice(stream);
    elemental_r_matrix->copyToDevice(stream);

    // determine nodes with common element
    auto common_element_matrix = std::make_shared<Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get nodes for element
        nodes = this->mesh()->elementNodes(element);

        // set system matrix elements
        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i) {
            for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
                (*common_element_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) = 1.0f;
            }
        }
    }

    // copy matrix to device
    common_element_matrix->copyToDevice(stream);

    // create sparse matrices
    this->s_matrix_ = std::make_shared<fastEIT::SparseMatrix>(common_element_matrix, stream);
    this->r_matrix_ = std::make_shared<fastEIT::SparseMatrix>(common_element_matrix, stream);

    // reduce matrices
    model::reduceMatrix(connectivity_matrix, this->s_matrix(), stream,
        this->connectivity_matrix());
    model::reduceMatrix(elemental_s_matrix, this->s_matrix(), stream,
        this->elemental_s_matrix());
    model::reduceMatrix(elemental_r_matrix, this->s_matrix(), stream,
        this->elemental_r_matrix());

    return common_element_matrix;
}

// init complete electrode model boundary conditions
template <
    class basis_function_type
>
std::tuple<std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>, std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>>>
    fastEIT::Model<basis_function_type>::initCEMMatrices(cudaStream_t stream) {
    // create matrices
    auto w_matrix = std::make_shared<Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->electrodes()->count(), stream);
    auto d_matrix = std::make_shared<Matrix<dtype::real>>(
        this->electrodes()->count(), this->electrodes()->count(), stream);

    // needed arrays
    std::array<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>, basis_function_type::nodes_per_edge> nodes;
    std::array<dtype::real, basis_function_type::nodes_per_edge> node_parameter;
    dtype::real integration_start, integration_end;

    // init z and w matrices
    for (dtype::index boundary_element = 0;
        boundary_element < this->mesh()->boundary()->rows();
        ++boundary_element) {
        // get boundary nodes
        nodes = this->mesh()->boundaryNodes(boundary_element);

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

        // calc node parameter centered to node 0
        for (dtype::size i = 0; i < basis_function_type::nodes_per_edge; ++i) {
            node_parameter[i] = math::circleParameter(std::get<1>(nodes[i]),
                parameter_offset);
        }

        for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
            // calc integration interval centered to node 0
            integration_start = math::circleParameter(
                std::get<0>(this->electrodes()->coordinates()[electrode]), parameter_offset);
            integration_end = math::circleParameter(
                std::get<1>(this->electrodes()->coordinates()[electrode]), parameter_offset);

            // intgrate if integration_start is left of integration_end
            if (integration_start < integration_end) {
                // calc z matrix element
                for (dtype::index i = 0; i < basis_function_type::nodes_per_edge; ++i) {
                    for (dtype::index j = 0; j < basis_function_type::nodes_per_edge; ++j) {
                        // calc z matrix element
                        if (this->source()->type() == "current") {
                            (*this->z_matrix())(std::get<0>(nodes[i]), std::get<0>(nodes[j])) +=
                                basis_function_type::integrateBoundaryEdgeWithBasis(
                                    node_parameter, i, j, integration_start, integration_end) /
                                this->electrodes()->impedance();
                        }
                    }

                    // calc w matrix element
                    if (this->source()->type() == "current") {
                        (*w_matrix)(std::get<0>(nodes[i]), electrode) -=
                            basis_function_type::integrateBoundaryEdge(
                                node_parameter, i, integration_start, integration_end) /
                            this->electrodes()->impedance();

                    } else if (this->source()->type() == "voltage") {
                        (*w_matrix)(std::get<0>(nodes[i]), electrode) -=
                            basis_function_type::integrateBoundaryEdge(
                                node_parameter, i, integration_start, integration_end) /
                            std::get<0>(this->electrodes()->shape());
                    }
                }
            }
        }
    }
    this->z_matrix()->copyToDevice(stream);

    // init d matrix
    if (this->source()->type() == "current") {
        for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
            (*d_matrix)(electrode, electrode) = std::get<0>(this->electrodes()->shape()) /
                this->electrodes()->impedance();
        }
    } else if (this->source()->type() == "voltage") {
        for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
            (*d_matrix)(electrode, electrode) = this->electrodes()->impedance() /
                std::get<0>(this->electrodes()->shape());
        }
    }

    return std::make_tuple(w_matrix, d_matrix);
}

// update model
template <
    class basis_function_type
>
void fastEIT::Model<basis_function_type>::update(const std::shared_ptr<Matrix<dtype::real>> gamma, cublasHandle_t handle,
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

    // create system matrices for all components
    dtype::real alpha = 0.0f;
    for (dtype::index component = 0; component < this->components_count(); ++component) {
        // calc alpha
        alpha = math::square(2.0f * component * M_PI / this->mesh()->height());

        // update system matrix
        modelKernel::updateSystemMatrix(this->s_matrix()->data_rows() / matrix::block_size, matrix::block_size, stream,
            this->s_matrix()->values(), this->r_matrix()->values(), this->s_matrix()->column_ids(),
            this->z_matrix()->device_data(), this->s_matrix()->density(), alpha,
            this->z_matrix()->data_rows(), this->system_matrix(component)->values());
    }
}

// calc excitation component
template <
    class basis_function_type
>
void fastEIT::Model<basis_function_type>::calcExcitationComponent(
    std::shared_ptr<Matrix<dtype::real>> excitation,
    dtype::size component, cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (excitation == nullptr) {
        throw std::invalid_argument("Model::calcNodalCurrentDensity: excitation == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Model::calcNodalCurrentDensity: handle == NULL");
    }

    // calc fourier coefficients for excitaion
    if (component == 0) {
        // calc ground mode
        excitation->scalarMultiply(1.0f / this->mesh()->height(), stream);
    } else {
        excitation->scalarMultiply(2.0f * sin(
            component * M_PI * std::get<1>(this->electrodes()->shape()) / this->mesh()->height()) /
            (component * M_PI * std::get<1>(this->electrodes()->shape())), stream);
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
