// fastEIT
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"
#include "fasteit/model_kernel.h"

// create solver model
template <
    class basis_function_type
>
fastEIT::Model<basis_function_type>::Model(
    std::shared_ptr<Mesh> mesh, std::shared_ptr<Electrodes> electrodes,
    dtype::real sigmaRef, dtype::size components_count, cublasHandle_t handle,
    cudaStream_t stream)
    : mesh_(mesh), electrodes_(electrodes), sigma_ref_(sigmaRef),
        components_count_(components_count) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("fastEIT::Model::Model: mesh == nullptr");
    }
    if (electrodes == nullptr) {
        throw std::invalid_argument("fastEIT::Model::Model: electrodes == nullptr");
    }
    if (components_count == 0) {
        throw std::invalid_argument("fastEIT::Model::Model: components_count == 0");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Model::Model: handle == nullptr");
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
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Model::init: handle == nullptr");
    }

    // init elemental matrices
    auto common_element_matrix = this->initElementalMatrices(stream);

    // create sparse matrices
    for (dtype::index component = 0; component < this->components_count(); ++component) {
        this->system_matrices_.push_back(std::make_shared<SparseMatrix<dtype::real>>(common_element_matrix, stream));
    }

    // create gamma
    auto gamma = std::make_shared<Matrix<dtype::real>>(this->mesh()->elements()->rows(), 1, stream);

    // update model
    this->update(gamma, handle, stream);
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

    std::vector<std::shared_ptr<Matrix<dtype::index>>> connectivity_matrices;
    std::vector<std::shared_ptr<Matrix<dtype::real>>> elemental_s_matrices,
        elemental_r_matrices;

    // fill intermediate connectivity and elemental matrices
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
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
        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; i++)
        for (dtype::index j = 0; j < basis_function_type::nodes_per_element; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            temp = (*element_count)(std::get<0>(nodes[i]), std::get<0>(nodes[j]));
            if (connectivity_matrices.size() <= temp) {
                connectivity_matrices.push_back(std::make_shared<Matrix<dtype::index>>(
                    this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream,
                    dtype::invalid_index));
                elemental_s_matrices.push_back(std::make_shared<Matrix<dtype::real>>(
                    this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream));
                elemental_r_matrices.push_back(std::make_shared<Matrix<dtype::real>>(
                    this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream));
            }

            // set connectivity element
            (*connectivity_matrices[temp])(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                element;

            // set elemental system element
            (*elemental_s_matrices[temp])(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                basis_functions[i]->integrateGradientWithBasis(basis_functions[j]);

            // set elemental residual element
            (*elemental_r_matrices[temp])(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                basis_functions[i]->integrateWithBasis(basis_functions[j]);

            // increment element count
            (*element_count)(std::get<0>(nodes[i]), std::get<0>(nodes[j]))++;
        }
    }

    // upload intermediate matrices
    for (dtype::index i = 0; i < connectivity_matrices.size(); ++i) {
        connectivity_matrices[i]->copyToDevice(stream);
        elemental_s_matrices[i]->copyToDevice(stream);
        elemental_r_matrices[i]->copyToDevice(stream);
    }

    // determine nodes with common element
    auto common_element_matrix = std::make_shared<Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get nodes for element
        nodes = this->mesh()->elementNodes(element);

        // set system matrix elements
        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i)
        for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
            (*common_element_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) = 1.0f;
        }
    }

    // copy matrix to device
    common_element_matrix->copyToDevice(stream);

    // create sparse matrices
    this->s_matrix_ = std::make_shared<fastEIT::SparseMatrix<dtype::real>>(common_element_matrix, stream);
    this->r_matrix_ = std::make_shared<fastEIT::SparseMatrix<dtype::real>>(common_element_matrix, stream);

    // create elemental matrices
    this->connectivity_matrix_ = std::make_shared<Matrix<dtype::index>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * connectivity_matrices.size(), stream, dtype::invalid_index);
    this->elemental_s_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * elemental_s_matrices.size(), stream);
    this->elemental_r_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        sparseMatrix::block_size * elemental_r_matrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    for (dtype::index i = 0; i < connectivity_matrices.size(); ++i) {
        model::reduceMatrix(connectivity_matrices[i], this->s_matrix(), i, stream,
            this->connectivity_matrix());
        model::reduceMatrix(elemental_s_matrices[i], this->s_matrix(), i, stream,
            this->elemental_s_matrix());
        model::reduceMatrix(elemental_r_matrices[i], this->s_matrix(), i, stream,
            this->elemental_r_matrix());
    }

    return common_element_matrix;
}

// update model
template <
    class basis_function_type
>
void fastEIT::Model<basis_function_type>::update(const std::shared_ptr<Matrix<dtype::real>> gamma, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::Model::init: handle == nullptr");
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
    for (dtype::index component = 0; component < this->components_count(); ++component) {
        // calc alpha
        alpha = fastEIT::math::square(2.0f * component * M_PI / this->mesh()->height());

        // init system matrix with 2d system matrix
        if (cublasScopy(handle, this->s_matrix()->data_rows() * sparseMatrix::block_size,
            this->s_matrix()->values(), 1, this->system_matrix(component)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "fastEIT::Model::update: calc system matrices for all harmonics");
        }

        // add alpha * residualMatrix
        if (cublasSaxpy(handle, this->s_matrix()->data_rows() * sparseMatrix::block_size, &alpha,
            this->r_matrix()->values(), 1, this->system_matrix(component)->values(), 1)
            != CUBLAS_STATUS_SUCCESS) {
            throw std::logic_error(
                "fastEIT::Model::update: calc system matrices for all harmonics");
        }
    }
}

// reduce matrix
template <
    class type
>
void fastEIT::model::reduceMatrix(const std::shared_ptr<Matrix<type>> intermediateMatrix,
    const std::shared_ptr<SparseMatrix<dtype::real>> shape, dtype::index offset,
    cudaStream_t stream, std::shared_ptr<Matrix<type>> matrix) {
    // check input
    if (intermediateMatrix == nullptr) {
        throw std::invalid_argument("fastEIT::model::reduceMatrix: intermediateMatrix == nullptr");
    }
    if (shape == nullptr) {
        throw std::invalid_argument("fastEIT::model::reduceMatrix: shape == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("fastEIT::model::reduceMatrix: matrix == nullptr");
    }

    // block size
    dim3 blocks(matrix->data_rows() / matrix::block_size, 1);
    dim3 threads(matrix::block_size, matrix::block_size);

    // reduce matrix
    modelKernel::reduceMatrix<type>(blocks, threads, stream,
        intermediateMatrix->device_data(), shape->column_ids(), matrix->data_rows(),
        offset, matrix->device_data());
}

// update matrix
void fastEIT::model::updateMatrix(const std::shared_ptr<Matrix<dtype::real>> elements,
    const std::shared_ptr<Matrix<dtype::real>> gamma,
    const std::shared_ptr<Matrix<dtype::index>> connectivityMatrix,
    dtype::real sigmaRef, cudaStream_t stream,
    std::shared_ptr<SparseMatrix<dtype::real>> matrix) {
    // check input
    if (elements == nullptr) {
        throw std::invalid_argument("fastEIT::model::updateMatrix: elements == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("fastEIT::model::updateMatrix: gamma == nullptr");
    }
    if (connectivityMatrix == nullptr) {
        throw std::invalid_argument("fastEIT::model::updateMatrix: connectivityMatrix == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("fastEIT::model::updateMatrix: matrix == nullptr");
    }

    // dimension
    dim3 threads(matrix::block_size, matrix::block_size);
    dim3 blocks(matrix->data_rows() / matrix::block_size, 1);

    // execute kernel
    modelKernel::updateMatrix(blocks, threads, stream,
        connectivityMatrix->device_data(), elements->device_data(), gamma->device_data(),
        sigmaRef, connectivityMatrix->data_rows(), connectivityMatrix->data_columns(), matrix->values());
}

// specialisation
template void fastEIT::model::reduceMatrix<fastEIT::dtype::real>(const std::shared_ptr<Matrix<fastEIT::dtype::real>>,
    const std::shared_ptr<SparseMatrix<dtype::real>>, fastEIT::dtype::index, cudaStream_t,
    std::shared_ptr<Matrix<fastEIT::dtype::real>>);
template void fastEIT::model::reduceMatrix<fastEIT::dtype::index>(const std::shared_ptr<Matrix<fastEIT::dtype::index>>,
    const std::shared_ptr<SparseMatrix<dtype::real>>, fastEIT::dtype::index, cudaStream_t,
    std::shared_ptr<Matrix<fastEIT::dtype::index>>);

template class fastEIT::Model<fastEIT::basis::Linear>;
template class fastEIT::Model<fastEIT::basis::Quadratic>;
