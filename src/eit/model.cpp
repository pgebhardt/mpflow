// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "mpflow/mpflow.h"
#include "mpflow/eit/model_kernel.h"

// 2.5D model base class
mpFlow::EIT::model::Base::Base(
    std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Electrodes> electrodes,
    std::shared_ptr<source::Source> source, dtype::real sigma_ref,
    dtype::size component_count)
    : mesh_(mesh), electrodes_(electrodes), source_(source), sigma_ref_(sigma_ref),
        component_count_(component_count) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::Base::Base: mesh == nullptr");
    }
    if (electrodes == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::Base::Base: electrodes == nullptr");
    }
    if (source == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::Base::Base: electrodes == nullptr");
    }
    if (component_count == 0) {
        throw std::invalid_argument("mpFlow::EIT::model::Base::Base: component_count == 0");
    }
    if (this->source()->component_count() != this->component_count()) {
        throw std::invalid_argument(
            "mpFlow::EIT::model::Base::Base: source.component_count != component_count");
    }
}

// create 2.5D model
template <
    class basis_function_type
>
mpFlow::EIT::Model<basis_function_type>::Model(
    std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Electrodes> electrodes,
    std::shared_ptr<source::Source> source, dtype::real sigma_ref,
    dtype::size component_count, cublasHandle_t handle, cudaStream_t stream)
    : model::Base(mesh, electrodes, source, sigma_ref, component_count) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Model::Model: handle == nullptr");
    }

    // create matrices
    this->jacobian_ = std::make_shared<numeric::Matrix<dtype::real>>(
        math::roundTo(this->source()->measurement_count(), numeric::matrix::block_size) *
        math::roundTo(this->source()->drive_count(), numeric::matrix::block_size),
        this->mesh()->elements()->rows(), stream);
    this->elemental_jacobian_matrix_ = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->elements()->rows(),
        math::square(basis_function_type::nodes_per_element), stream);
    for (dtype::index component = 0; component < this->component_count(); ++component) {
        this->potential_.push_back(std::make_shared<numeric::Matrix<dtype::real>>(
            this->mesh()->nodes()->rows() + this->electrodes()->count(),
            this->source()->drive_count() + this->source()->measurement_count(), stream));
    }

    // init model
    this->init(handle, stream);

    // init jacobian calculation matrix
    this->initJacobianCalculationMatrix(handle, stream);
}

// init model
template <
    class basis_function_type
>
void mpFlow::EIT::Model<basis_function_type>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Model::init: handle == nullptr");
    }

    // init elemental matrices
    auto common_element_matrix = this->initElementalMatrices(stream);

    // assamble initial system matrices
    auto system_matrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows() + this->electrodes()->count(),
        this->mesh()->nodes()->rows() + this->electrodes()->count(),
        stream);

    // fill s + r + z
    for (dtype::index row = 0; row < this->mesh()->nodes()->rows(); ++row)
    for (dtype::index column = 0; column < this->mesh()->nodes()->rows(); ++column) {
        (*system_matrix)(row, column) = (*common_element_matrix)(row, column);
    }

    // fill w and x
    for (dtype::index node = 0; node < this->mesh()->nodes()->rows(); ++node)
    for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
        (*system_matrix)(node, electrode + this->mesh()->nodes()->rows()) =
            (*this->source()->w_matrix())(node, electrode);

        (*system_matrix)(electrode + this->mesh()->nodes()->rows(), node) =
            (*this->source()->x_matrix())(electrode, node);
    }

    // fill d matrix
    for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
        (*system_matrix)(electrode + this->mesh()->nodes()->rows(),
            electrode + this->mesh()->nodes()->rows()) =
            (*this->source()->d_matrix())(electrode, electrode);
    }

    // create sparse matrices
    system_matrix->copyToDevice(stream);
    for (dtype::index component = 0; component < this->component_count(); ++component) {
        this->system_matrices_.push_back(std::make_shared<numeric::SparseMatrix<dtype::real>>(
            system_matrix, stream));
    }

    // create gamma
    auto gamma = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->elements()->rows(), 1, stream);

    // update model
    this->update(gamma, stream);
}

// init elemental matrices
template <
    class basis_function_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::Model<basis_function_type>::initElementalMatrices(
    cudaStream_t stream) {
    // create intermediate matrices
    std::vector<std::vector<dtype::index>> element_count(
        this->mesh()->nodes()->rows(), std::vector<dtype::index>(
        this->mesh()->nodes()->rows(), 0));
    std::vector<std::vector<std::vector<dtype::index>>> connectivity_matrices;
    std::vector<std::vector<std::vector<dtype::real>>> elemental_s_matrices,
        elemental_r_matrices;

    // fill intermediate connectivity and elemental matrices
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
    std::array<std::tuple<dtype::real, dtype::real>,
        basis_function_type::nodes_per_element> nodes_coordinates;
    std::array<std::shared_ptr<basis_function_type>,
        basis_function_type::nodes_per_element> basis_functions;
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
            auto level = element_count[std::get<0>(nodes[i])][std::get<0>(nodes[j])];
            if (connectivity_matrices.size() <= level) {
                connectivity_matrices.push_back(std::vector<std::vector<dtype::index>>(
                    this->mesh()->nodes()->rows(), std::vector<dtype::index>(
                    this->mesh()->nodes()->rows(), dtype::invalid_index)));
                elemental_s_matrices.push_back(std::vector<std::vector<dtype::real>>(
                    this->mesh()->nodes()->rows(), std::vector<dtype::real>(
                    this->mesh()->nodes()->rows(), 0.0)));
                elemental_r_matrices.push_back(std::vector<std::vector<dtype::real>>(
                    this->mesh()->nodes()->rows(), std::vector<dtype::real>(
                    this->mesh()->nodes()->rows(), 0.0)));
            }

            // set connectivity element
            connectivity_matrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                element;

            // set elemental system element
            elemental_s_matrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                basis_functions[i]->integrateGradientWithBasis(basis_functions[j]);

            // set elemental residual element
            elemental_r_matrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                basis_functions[i]->integrateWithBasis(basis_functions[j]);

            // increment element count
            element_count[std::get<0>(nodes[i])][std::get<0>(nodes[j])]++;
        }
    }

    // determine nodes with common element
    auto common_element_matrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        nodes = this->mesh()->elementNodes(element);

        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i)
        for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
            (*common_element_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) = 1.0f;
        }
    }
    common_element_matrix->copyToDevice(stream);

    // create sparse matrices
    this->s_matrix_ = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        common_element_matrix, stream);
    this->r_matrix_ = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        common_element_matrix, stream);

    // create elemental matrices
    this->connectivity_matrix_ = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh()->nodes()->rows(),
        numeric::sparseMatrix::block_size * connectivity_matrices.size(), stream, dtype::invalid_index);
    this->elemental_s_matrix_ = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        numeric::sparseMatrix::block_size * elemental_s_matrices.size(), stream);
    this->elemental_r_matrix_ = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        numeric::sparseMatrix::block_size * elemental_r_matrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    auto connectivity_matrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream,
        dtype::invalid_index);
    auto elemental_s_matrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    auto elemental_r_matrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    for (dtype::index level = 0; level < connectivity_matrices.size(); ++level) {
        for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
            // get element nodes
            nodes = this->mesh()->elementNodes(element);

            for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i)
            for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
                (*connectivity_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    connectivity_matrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
                (*elemental_s_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    elemental_s_matrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
                (*elemental_r_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    elemental_r_matrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
            }
        }
        connectivity_matrix->copyToDevice(stream);
        elemental_s_matrix->copyToDevice(stream);
        elemental_r_matrix->copyToDevice(stream);
        cudaStreamSynchronize(stream);

        model::reduceMatrix(connectivity_matrix, this->s_matrix(), level, stream,
            this->connectivity_matrix());
        model::reduceMatrix(elemental_s_matrix, this->s_matrix(), level, stream,
            this->elemental_s_matrix());
        model::reduceMatrix(elemental_r_matrix, this->r_matrix(), level, stream,
            this->elemental_r_matrix());
    }

    return common_element_matrix;
}

// init jacobian calculation matrix
template <
    class basis_function_type
>
void mpFlow::EIT::Model<basis_function_type>::initJacobianCalculationMatrix(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument(
            "mpFlow::EIT::Model::initJacobianCalculationMatrix: handle == nullptr");
    }

    // variables
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
    std::array<std::tuple<dtype::real, dtype::real>,
       basis_function_type::nodes_per_element> nodes_coordinates;
    std::array<std::shared_ptr<basis_function_type>,
        basis_function_type::nodes_per_element> basis_functions;

    // fill connectivity and elementalJacobianMatrix
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get element nodes
        nodes = this->mesh()->elementNodes(element);

        // extract nodes coordinates
        for (dtype::index node = 0; node < basis_function_type::nodes_per_element; ++node) {
            nodes_coordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < basis_function_type::nodes_per_element; ++node) {
            basis_functions[node] = std::make_shared<basis_function_type>(
                nodes_coordinates, node);
        }

        // fill matrix
        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i)
        for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
            // set elementalJacobianMatrix element
            (*this->elemental_jacobian_matrix())(element, i +
                j * basis_function_type::nodes_per_element) =
                basis_functions[i]->integrateGradientWithBasis(basis_functions[j]);
        }
    }
    this->elemental_jacobian_matrix()->copyToDevice(stream);
}

// calc jacobian
template <
    class basis_function_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::EIT::Model<basis_function_type>::calcJacobian(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, cudaStream_t stream) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::Model::calcJacobian: gamma == nullptr");
    }

    // calc jacobian
    model::calcJacobian<basis_function_type>(
        gamma, this->potential(0), this->mesh()->elements(),
        this->elemental_jacobian_matrix(), this->source()->drive_count(),
        this->source()->measurement_count(), this->sigma_ref(),
        false, stream, this->jacobian());
    for (dtype::index component = 1; component < this->component_count(); ++component) {
        model::calcJacobian<basis_function_type>(
            gamma, this->potential(component), this->mesh()->elements(),
            this->elemental_jacobian_matrix(), this->source()->drive_count(),
            this->source()->measurement_count(), this->sigma_ref(),
            true, stream, this->jacobian());
    }

    // switch sign if current source
    if (this->source()->type() == "current") {
        this->jacobian()->scalarMultiply(-1.0, stream);
    }

    return this->jacobian();
}

// update model
template <
    class basis_function_type
>
void mpFlow::EIT::Model<basis_function_type>::update(
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma, cudaStream_t stream) {
    // update matrices
    model::updateMatrix(this->elemental_s_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->s_matrix());
    model::updateMatrix(this->elemental_r_matrix(), gamma, this->connectivity_matrix(),
        this->sigma_ref(), stream, this->r_matrix());

    // create system matrices for all harmonics
    dtype::real alpha = 0.0f;
    for (dtype::index component = 0; component < this->component_count(); ++component) {
        // calc alpha
        alpha = math::square(2.0f * component * M_PI / this->mesh()->height());

        // update system matrix
        modelKernel::updateSystemMatrix(this->s_matrix()->data_rows() / numeric::matrix::block_size,
            numeric::matrix::block_size, stream,
            this->s_matrix()->values(), this->r_matrix()->values(), this->s_matrix()->column_ids(),
            this->source()->z_matrix()->device_data(), this->s_matrix()->density(), alpha,
            this->source()->z_matrix()->data_rows(), this->system_matrix(component)->values());
    }
}

// reduce matrix
template <
    class type
>
void mpFlow::EIT::model::reduceMatrix(const std::shared_ptr<numeric::Matrix<type>> intermediateMatrix,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>> shape, dtype::index offset,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<type>> matrix) {
    // check input
    if (intermediateMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::reduceMatrix: intermediateMatrix == nullptr");
    }
    if (shape == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::reduceMatrix: shape == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::reduceMatrix: matrix == nullptr");
    }

    // block size
    dim3 blocks(matrix->data_rows() / numeric::matrix::block_size, 1);
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);

    // reduce matrix
    modelKernel::reduceMatrix<type>(blocks, threads, stream,
        intermediateMatrix->device_data(), shape->column_ids(), matrix->data_rows(),
        offset, matrix->device_data());
}

// update matrix
void mpFlow::EIT::model::updateMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> elements,
    const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
    const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix,
    dtype::real sigmaRef, cudaStream_t stream,
    std::shared_ptr<numeric::SparseMatrix<dtype::real>> matrix) {
    // check input
    if (elements == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::updateMatrix: elements == nullptr");
    }
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::updateMatrix: gamma == nullptr");
    }
    if (connectivityMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::updateMatrix: connectivityMatrix == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::updateMatrix: matrix == nullptr");
    }

    // dimension
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);
    dim3 blocks(matrix->data_rows() / numeric::matrix::block_size, 1);

    // execute kernel
    modelKernel::updateMatrix(blocks, threads, stream,
        connectivityMatrix->device_data(), elements->device_data(), gamma->device_data(),
        sigmaRef, connectivityMatrix->data_rows(), connectivityMatrix->data_columns(), matrix->values());
}

// calc jacobian
template <
    class basis_function_type
>
void mpFlow::EIT::model::calcJacobian(const std::shared_ptr<numeric::Matrix<dtype::real>> gamma,
    const std::shared_ptr<numeric::Matrix<dtype::real>> potential,
    const std::shared_ptr<numeric::Matrix<dtype::index>> elements,
    const std::shared_ptr<numeric::Matrix<dtype::real>> elemental_jacobian_matrix,
    dtype::size drive_count, dtype::size measurment_count, dtype::real sigma_ref,
    bool additiv, cudaStream_t stream, std::shared_ptr<numeric::Matrix<dtype::real>> jacobian) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::calcJacobian: gamma == nullptr");
    }
    if (potential == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::calcJacobian: potential == nullptr");
    }
    if (elements == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::calcJacobian: elements == nullptr");
    }
    if (elemental_jacobian_matrix == nullptr) {
        throw std::invalid_argument(
        "mpFlow::EIT::model::calcJacobian: elemental_jacobian_matrix == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("mpFlow::EIT::model::calcJacobian: jacobian == nullptr");
    }

    // dimension
    dim3 blocks(jacobian->data_rows() / numeric::matrix::block_size,
        jacobian->data_columns() / numeric::matrix::block_size);
    dim3 threads(numeric::matrix::block_size, numeric::matrix::block_size);

    // calc jacobian
    modelKernel::calcJacobian<basis_function_type::nodes_per_element>(blocks, threads, stream,
        potential->device_data(), &potential->device_data()[drive_count * potential->data_rows()],
        elements->device_data(), elemental_jacobian_matrix->device_data(),
        gamma->device_data(), sigma_ref, jacobian->data_rows(), jacobian->data_columns(),
        potential->data_rows(), elements->rows(), drive_count, measurment_count, additiv,
        jacobian->device_data());
}

// specialisation
template void mpFlow::EIT::model::reduceMatrix<mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::EIT::model::reduceMatrix<mpFlow::dtype::index>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);

template void mpFlow::EIT::model::calcJacobian<mpFlow::FEM::basis::Linear>(
    const std::shared_ptr<numeric::Matrix<dtype::real>>, const std::shared_ptr<numeric::Matrix<dtype::real>>,
    const std::shared_ptr<numeric::Matrix<dtype::index>>, const std::shared_ptr<numeric::Matrix<dtype::real>>,
    dtype::size, dtype::size, dtype::real, bool, cudaStream_t, std::shared_ptr<numeric::Matrix<dtype::real>>);
template void mpFlow::EIT::model::calcJacobian<mpFlow::FEM::basis::Quadratic>(
    const std::shared_ptr<numeric::Matrix<dtype::real>>, const std::shared_ptr<numeric::Matrix<dtype::real>>,
    const std::shared_ptr<numeric::Matrix<dtype::index>>, const std::shared_ptr<numeric::Matrix<dtype::real>>,
    dtype::size, dtype::size, dtype::real, bool, cudaStream_t, std::shared_ptr<numeric::Matrix<dtype::real>>);

template class mpFlow::EIT::Model<mpFlow::FEM::basis::Linear>;
template class mpFlow::EIT::Model<mpFlow::FEM::basis::Quadratic>;
