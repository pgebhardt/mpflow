// mpFlow
//
// Copyright (C) 2014  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "mpflow/mpflow.h"
#include "mpflow/uwb/model_kernel.h"

// UWB model base class
mpFlow::UWB::model::Base::Base(
    std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Windows> windows)
    : _mesh(mesh), _windows(windows)  {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::Base::Base: mesh == nullptr");
    }
    if (windows == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::Base::Base: windows == nullptr");
    }
}

// create uwb model
template <
    class basis_function_type
>
mpFlow::UWB::Model<basis_function_type>::Model(
    std::shared_ptr<numeric::IrregularMesh> mesh, std::shared_ptr<Windows> windows,
    cublasHandle_t handle, cudaStream_t stream)
    : model::Base(mesh, windows) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::Model::Model: handle == nullptr");
    }

    // create matrices
    // TODO: set columns according to source
    this->_fieldReal = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), 1, stream);
    this->_fieldImaginary = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), 1, stream);

    // init model
    this->init(handle, stream);
}

// init model
template <
    class basis_function_type
>
void mpFlow::UWB::Model<basis_function_type>::init(cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::Model::init: handle == nullptr");
    }

    // init elemental matrices
    auto commonElementMatrix = this->initElementalMatrices(stream);

    // assamble initial system matrix
    auto systemMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        2.0 * this->mesh()->nodes()->rows(), 2.0 * this->mesh()->nodes()->rows(), stream);
    for (dtype::index row = 0; row < this->mesh()->nodes()->rows(); ++row)
    for (dtype::index column = 0; column < this->mesh()->nodes()->rows(); ++column) {
        (*systemMatrix)(row, column) = (*commonElementMatrix)(row, column);
        (*systemMatrix)(row + this->mesh()->nodes()->rows(), column) = (*commonElementMatrix)(row, column);
        (*systemMatrix)(row, column + this->mesh()->nodes()->rows()) = (*commonElementMatrix)(row, column);
        (*systemMatrix)(row + this->mesh()->nodes()->rows(), column + this->mesh()->nodes()->rows()) =
            (*commonElementMatrix)(row, column);
    }

    // create sparse matrix
    systemMatrix->copyToDevice(stream);
    this->_systemMatrix = std::make_shared<numeric::SparseMatrix<dtype::real>>(
        systemMatrix, stream);

    // create gamma
    // TODO: correct definition of real and imaginary part
    auto epsilonR = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->elements()->rows(), 1, stream, 1.0);
    auto sigma = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->elements()->rows(), 1, stream);

    // init s matrix only once and update rest of model
    model::updateMatrix(this->elementalSMatrix(), epsilonR, this->connectivityMatrix(),
        stream, this->sMatrix());
    this->update(epsilonR, sigma, stream);
}

// init elemental matrices
template <
    class basis_function_type
>
std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>
    mpFlow::UWB::Model<basis_function_type>::initElementalMatrices(
    cudaStream_t stream) {
    // create intermediate matrices
    std::vector<std::vector<dtype::index>> elementCount(
        this->mesh()->nodes()->rows(), std::vector<dtype::index>(
        this->mesh()->nodes()->rows(), 0));
    std::vector<std::vector<std::vector<dtype::index>>> connectivityMatrices;
    std::vector<std::vector<std::vector<dtype::real>>> elementalSMatrices,
        elementalRMatrices;

    // fill intermediate connectivity and elemental matrices
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
    std::array<std::tuple<dtype::real, dtype::real>,
        basis_function_type::nodes_per_element> nodesCoordinates;
    std::array<std::shared_ptr<basis_function_type>,
        basis_function_type::nodes_per_element> basisFunction;
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        // get element nodes
        nodes = this->mesh()->elementNodes(element);

        // extract coordinates
        for (dtype::index node = 0; node < basis_function_type::nodes_per_element; ++node) {
            nodesCoordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < basis_function_type::nodes_per_element; ++node) {
            basisFunction[node] = std::make_shared<basis_function_type>(
                nodesCoordinates, node);
        }

        // set connectivity and elemental residual matrix elements
        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; i++)
        for (dtype::index j = 0; j < basis_function_type::nodes_per_element; j++) {
            // get current element count and add new intermediate matrices if 
            // neccessary
            auto level = elementCount[std::get<0>(nodes[i])][std::get<0>(nodes[j])];
            if (connectivityMatrices.size() <= level) {
                connectivityMatrices.push_back(std::vector<std::vector<dtype::index>>(
                    this->mesh()->nodes()->rows(), std::vector<dtype::index>(
                    this->mesh()->nodes()->rows(), dtype::invalid_index)));
                elementalSMatrices.push_back(std::vector<std::vector<dtype::real>>(
                    this->mesh()->nodes()->rows(), std::vector<dtype::real>(
                    this->mesh()->nodes()->rows(), 0.0)));
                elementalRMatrices.push_back(std::vector<std::vector<dtype::real>>(
                    this->mesh()->nodes()->rows(), std::vector<dtype::real>(
                    this->mesh()->nodes()->rows(), 0.0)));
            }

            // set connectivity element
            connectivityMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                element;

            // set elemental system element
            elementalSMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                basisFunction[i]->integrateGradientWithBasis(basisFunction[j]);

            // set elemental residual element
            elementalRMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])] =
                basisFunction[i]->integrateWithBasis(basisFunction[j]);

            // increment element count
            elementCount[std::get<0>(nodes[i])][std::get<0>(nodes[j])]++;
        }
    }

    // determine nodes with common element
    auto commonElementMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
        nodes = this->mesh()->elementNodes(element);

        for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i)
        for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
            (*commonElementMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) = 1.0f;
        }
    }
    commonElementMatrix->copyToDevice(stream);

    // create sparse matrices
    this->_sMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        commonElementMatrix, stream);
    this->_rMatrix = std::make_shared<mpFlow::numeric::SparseMatrix<dtype::real>>(
        commonElementMatrix, stream);

    // create elemental matrices
    this->_connectivityMatrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh()->nodes()->rows(),
        numeric::sparseMatrix::block_size * connectivityMatrices.size(), stream, dtype::invalid_index);
    this->_elementalSMatrix = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        numeric::sparseMatrix::block_size * elementalSMatrices.size(), stream);
    this->_elementalRMatrix = std::make_shared<numeric::Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        numeric::sparseMatrix::block_size * elementalRMatrices.size(), stream);

    // store all elemental matrices in one matrix for each type in a sparse
    // matrix like format
    auto connectivity_matrix = std::make_shared<numeric::Matrix<dtype::index>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream,
        dtype::invalid_index);
    auto elementalSMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    auto elementalRMatrix = std::make_shared<numeric::Matrix<dtype::real>>(
        this->mesh()->nodes()->rows(), this->mesh()->nodes()->rows(), stream);
    for (dtype::index level = 0; level < connectivityMatrices.size(); ++level) {
        for (dtype::index element = 0; element < this->mesh()->elements()->rows(); ++element) {
            // get element nodes
            nodes = this->mesh()->elementNodes(element);

            for (dtype::index i = 0; i < basis_function_type::nodes_per_element; ++i)
            for (dtype::index j = 0; j < basis_function_type::nodes_per_element; ++j) {
                (*connectivity_matrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    connectivityMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
                (*elementalSMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    elementalSMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
                (*elementalRMatrix)(std::get<0>(nodes[i]), std::get<0>(nodes[j])) =
                    elementalRMatrices[level][std::get<0>(nodes[i])][std::get<0>(nodes[j])];
            }
        }
        connectivity_matrix->copyToDevice(stream);
        elementalSMatrix->copyToDevice(stream);
        elementalRMatrix->copyToDevice(stream);
        cudaStreamSynchronize(stream);

        model::reduceMatrix(connectivity_matrix, this->sMatrix(), level, stream,
            this->connectivityMatrix());
        model::reduceMatrix(elementalSMatrix, this->sMatrix(), level, stream,
            this->elementalSMatrix());
        model::reduceMatrix(elementalRMatrix, this->rMatrix(), level, stream,
            this->elementalRMatrix());
    }

    return commonElementMatrix;
}

// update model
template <
    class basis_function_type
>
void mpFlow::UWB::Model<basis_function_type>::update(
    const std::shared_ptr<numeric::Matrix<dtype::real>> epsilonR,
    const std::shared_ptr<numeric::Matrix<dtype::real>> sigma, cudaStream_t stream) {
    // TODO: no fixed frequency
    dtype::real frequency = 1e9;

    // calc each quadrant of system matrix
    // top left
    model::updateMatrix(this->elementalRMatrix(), epsilonR, this->connectivityMatrix(), stream, this->rMatrix());
    modelKernel::updateSystemMatrix(this->sMatrix()->data_rows() / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix()->values(), this->rMatrix()->values(),
        this->sMatrix()->column_ids(), this->sMatrix()->density(), 1.0,
        math::square(2.0 * M_PI * frequency) * constants::epsilon0 * constants::mu0,
        0, 0, this->systemMatrix()->values());

    // bottom right
    modelKernel::updateSystemMatrix(this->sMatrix()->data_rows() / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix()->values(), this->rMatrix()->values(),
        this->sMatrix()->column_ids(), this->sMatrix()->density(), 1.0,
        math::square(2.0 * M_PI * frequency) * constants::epsilon0 * constants::mu0,
        this->sMatrix()->data_rows(), this->sMatrix()->data_columns(), this->systemMatrix()->values());

    // top right
    model::updateMatrix(this->elementalRMatrix(), sigma, this->connectivityMatrix(), stream, this->rMatrix());
    modelKernel::updateSystemMatrix(this->sMatrix()->data_rows() / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix()->values(), this->rMatrix()->values(),
        this->sMatrix()->column_ids(), this->sMatrix()->density(), 0.0,
        -2.0 * M_PI * frequency * constants::mu0,
        0, this->sMatrix()->data_rows(), this->systemMatrix()->values());

    // bottom left
    modelKernel::updateSystemMatrix(this->sMatrix()->data_rows() / numeric::matrix::block_size,
        numeric::matrix::block_size, stream, this->sMatrix()->values(), this->rMatrix()->values(),
        this->sMatrix()->column_ids(), this->sMatrix()->density(), 0.0,
        2.0 * M_PI * frequency * constants::mu0,
        this->sMatrix()->data_rows(), 0, this->systemMatrix()->values());
}

// reduce matrix
template <
    class type
>
void mpFlow::UWB::model::reduceMatrix(const std::shared_ptr<numeric::Matrix<type>> intermediateMatrix,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>> shape, dtype::index offset,
    cudaStream_t stream, std::shared_ptr<numeric::Matrix<type>> matrix) {
    // check input
    if (intermediateMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::reduceMatrix: intermediateMatrix == nullptr");
    }
    if (shape == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::reduceMatrix: shape == nullptr");
    }
    if (matrix == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::reduceMatrix: matrix == nullptr");
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
void mpFlow::UWB::model::updateMatrix(const std::shared_ptr<numeric::Matrix<dtype::real>> elements,
    const std::shared_ptr<numeric::Matrix<dtype::real>> material,
    const std::shared_ptr<numeric::Matrix<dtype::index>> connectivityMatrix, cudaStream_t stream,
    std::shared_ptr<numeric::SparseMatrix<dtype::real>> result) {
    // check input
    if (elements == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::updateMatrix: elements == nullptr");
    }
    if (material == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::updateMatrix: material == nullptr");
    }
    if (connectivityMatrix == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::updateMatrix: connectivityMatrix == nullptr");
    }
    if (result == nullptr) {
        throw std::invalid_argument("mpFlow::UWB::model::updateMatrix: result == nullptr");
    }

    // dimension
    dim3 threads(numeric::matrix::block_size, numeric::sparseMatrix::block_size);
    dim3 blocks(material->data_rows() / numeric::matrix::block_size, 1);

    // execute kernel
    modelKernel::updateMatrix(blocks, threads, stream,
        connectivityMatrix->device_data(), elements->device_data(), material->device_data(),
        connectivityMatrix->data_rows(), connectivityMatrix->data_columns(), result->values());
}

// specialisation
template void mpFlow::UWB::model::reduceMatrix<mpFlow::dtype::real>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::real>>);
template void mpFlow::UWB::model::reduceMatrix<mpFlow::dtype::index>(
    const std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>,
    const std::shared_ptr<numeric::SparseMatrix<dtype::real>>, mpFlow::dtype::index, cudaStream_t,
    std::shared_ptr<numeric::Matrix<mpFlow::dtype::index>>);

template class mpFlow::UWB::Model<mpFlow::FEM::basis::Linear>;
template class mpFlow::UWB::Model<mpFlow::FEM::basis::Quadratic>;
