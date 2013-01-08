// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"
#include "../include/forward_kernel.h"

// create forward_solver
template <
    class numeric_solver_type,
    class model_type,
    class source_type
>
fastEIT::ForwardSolver<numeric_solver_type, model_type, source_type>::ForwardSolver(
    std::shared_ptr<model_type> model, std::shared_ptr<source_type> source,
    cublasHandle_t handle, cudaStream_t stream)
    : model_(model), source_(source) {
    // check input
    if (model == nullptr) {
        throw std::invalid_argument("ForwardSolver::ForwardSolver: model == nullptr");
    }
    if (source == nullptr) {
        throw std::invalid_argument("ForwardSolver::ForwardSolver: source == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("ForwardSolver::ForwardSolver: handle == NULL");
    }

    // create numeric_solver_type solver
    this->numeric_solver_ = std::make_shared<numeric_solver_type>(this->model()->mesh()->nodes()->rows(),
        this->model()->electrodes()->drive_count() + this->model()->electrodes()->measurement_count(), stream);

    // create matrices
    this->jacobian_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->electrodes()->measurement_pattern()->data_columns() *
        this->model()->electrodes()->drive_pattern()->data_columns(),
        this->model()->mesh()->elements()->rows(), stream);
    this->voltage_ = std::make_shared<Matrix<dtype::real>>(this->model()->electrodes()->measurement_count(),
        this->model()->electrodes()->drive_count(), stream);
    this->voltage_calculation_ = std::make_shared<Matrix<dtype::real>>(this->model()->electrodes()->measurement_count(),
        this->model()->mesh()->nodes()->rows(), stream);
    this->elemental_jacobian_matrix_ = std::make_shared<Matrix<dtype::real>>(this->model()->mesh()->elements()->rows(),
        math::square(model_type::basis_function_type::nodes_per_element), stream);

    // create matrices
    for (dtype::index component = 0; component < this->model()->components_count() + 1; ++component) {
        this->potential_.push_back(std::make_shared<Matrix<dtype::real>>(this->model()->mesh()->nodes()->rows(),
            this->model()->electrodes()->drive_count() + this->model()->electrodes()->measurement_count(), stream));
        this->current_density_.push_back(std::make_shared<Matrix<dtype::real>>(this->model()->mesh()->nodes()->rows(),
            this->model()->electrodes()->drive_count() + this->model()->electrodes()->measurement_count(), stream));
    }

    // create pattern matrix
    auto pattern = std::make_shared<Matrix<dtype::real>>(this->model()->electrodes()->count(),
        this->model()->electrodes()->drive_count() + this->model()->electrodes()->measurement_count(), stream);

    // fill pattern matrix with drive pattern
    for (dtype::index row = 0; row < pattern->rows(); ++row) {
        for (dtype::index column = 0; column < this->model()->electrodes()->drive_count(); ++column) {
            (*pattern)(row, column) = (*this->model()->electrodes()->drive_pattern())(row, column);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::index row = 0; row < pattern->rows(); ++row) {
        for (dtype::index column = 0; column < this->model()->electrodes()->measurement_count(); ++column) {
            (*pattern)(row, column + this->model()->electrodes()->drive_count()) =
                -(*this->model()->electrodes()->measurement_pattern())(row, column);
        }
    }
    pattern->copyToDevice(stream);

    // turn sign of pattern for correct current direction
    pattern->scalarMultiply(-1.0, stream);

    // calc excitation components
    for (dtype::index component = 0; component < this->model()->components_count() + 1; ++component) {
        this->model()->calcNodalCurrentDensity(pattern, component, handle, stream, this->current_density(component));
    }

    // calc voltage calculation matrix
    dtype::real alpha = 1.0, beta = 0.0;

    // one prerun for cublas
    cublasSetStream(handle, stream);
    cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        this->model()->electrodes()->measurement_pattern()->data_columns(),
        this->model()->excitation_matrix()->data_rows(),
        this->model()->electrodes()->measurement_pattern()->data_rows(), &alpha,
        this->model()->electrodes()->measurement_pattern()->device_data(),
        this->model()->electrodes()->measurement_pattern()->data_rows(),
        this->model()->excitation_matrix()->device_data(),
        this->model()->excitation_matrix()->data_rows(),
        &beta, this->voltage_calculation()->device_data(),
        this->voltage_calculation()->data_rows());

    if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T,
        this->model()->electrodes()->measurement_pattern()->data_columns(),
        this->model()->excitation_matrix()->data_rows(),
        this->model()->electrodes()->measurement_pattern()->data_rows(), &alpha,
        this->model()->electrodes()->measurement_pattern()->device_data(),
        this->model()->electrodes()->measurement_pattern()->data_rows(),
        this->model()->excitation_matrix()->device_data(),
        this->model()->excitation_matrix()->data_rows(),
        &beta, this->voltage_calculation()->device_data(),
        this->voltage_calculation()->data_rows())
        != CUBLAS_STATUS_SUCCESS) {
        throw std::logic_error("ForwardSolver::ForwardSolver: calc voltage calculation");
    }

    // init jacobian calculation matrix
    this->initJacobianCalculationMatrix(handle, stream);
}

// init jacobian calculation matrix
template <
    class numeric_solver_type,
    class model_type,
    class source_type
>
void fastEIT::ForwardSolver<numeric_solver_type, model_type, source_type>::initJacobianCalculationMatrix(
    cublasHandle_t handle, cudaStream_t stream) {
    // check input
    if (handle == NULL) {
        throw std::invalid_argument("ForwardSolver::initJacobianCalculationMatrix: handle == NULL");
    }

    // variables
    std::array<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>,
        model_type::basis_function_type::nodes_per_element> nodes;
    std::array<std::tuple<dtype::real, dtype::real>, model_type::basis_function_type::nodes_per_element> nodes_coordinates;
    std::array<std::shared_ptr<typename model_type::basis_function_type>,
        model_type::basis_function_type::nodes_per_element> basis_functions;

    // fill connectivity and elementalJacobianMatrix
    for (dtype::index element = 0; element < this->model()->mesh()->elements()->rows(); ++element) {
        // get element nodes
        nodes = this->model()->mesh()->elementNodes(element);

        // extract nodes coordinates
        for (dtype::index node = 0; node < model_type::basis_function_type::nodes_per_element; ++node) {
            nodes_coordinates[node] = std::get<1>(nodes[node]);
        }

        // calc corresponding basis functions
        for (dtype::index node = 0; node < model_type::basis_function_type::nodes_per_element; ++node) {
            basis_functions[node] = std::make_shared<typename model_type::basis_function_type>(
                nodes_coordinates, node);
        }

        // fill matrix
        for (dtype::index i = 0; i < model_type::basis_function_type::nodes_per_element; ++i) {
            for (dtype::index j = 0; j < model_type::basis_function_type::nodes_per_element; ++j) {
                // set elementalJacobianMatrix element
                (*this->elemental_jacobian_matrix())(element, i + j * model_type::basis_function_type::nodes_per_element) =
                    basis_functions[i]->integrateGradientWithBasis(basis_functions[j]);
            }
        }
    }

    // upload to device
    this->elemental_jacobian_matrix()->copyToDevice(stream);
}

// forward solving
template <
    class numeric_solver_type,
    class model_type,
    class source_type
>
std::shared_ptr<fastEIT::Matrix<fastEIT::dtype::real>> fastEIT::ForwardSolver<numeric_solver_type, model_type, source_type>::solve(
    const std::shared_ptr<Matrix<dtype::real>> gamma, dtype::size steps, cublasHandle_t handle,
    cudaStream_t stream) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("ForwardSolver::solve: gamma == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("ForwardSolver::solve: handle == NULL");
    }

    // update system matrix
    this->model()->update(gamma, handle, stream);

    // solve for ground mode
    this->numeric_solver()->solve(this->model()->system_matrix(0), this->current_density(0),
        steps, true, stream, this->potential(0));

    // solve for higher harmonics
    for (dtype::index component = 1; component < this->model()->components_count(); ++component) {
        this->numeric_solver()->solve(this->model()->system_matrix(component), this->current_density(component),
            steps, false, stream, this->potential(component));
    }

    // calc jacobian
    forward::calcJacobian<model_type>(gamma, this->potential(0),
        this->model()->mesh()->elements(), this->elemental_jacobian_matrix(),
        this->model()->electrodes()->drive_count(), this->model()->electrodes()->measurement_count(),
        this->model()->sigma_ref(), false, stream, this->jacobian());
    for (dtype::index component = 1; component < this->model()->components_count(); ++component) {
        forward::calcJacobian<model_type>(gamma, this->potential(component),
            this->model()->mesh()->elements(), this->elemental_jacobian_matrix(),
            this->model()->electrodes()->drive_count(), this->model()->electrodes()->measurement_count(),
            this->model()->sigma_ref(), true, stream, this->jacobian());
    }

    // turn sign of jacobian, because of voltage jacobian
    this->jacobian()->scalarMultiply(-1.0, stream);

    // set stream
    cublasSetStream(handle, stream);

    // add voltage
    dtype::real alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltage_calculation()->data_rows(),
        this->model()->electrodes()->drive_count(), this->voltage_calculation()->data_columns(), &alpha,
        this->voltage_calculation()->device_data(), this->voltage_calculation()->data_rows(),
        this->potential(0)->device_data(), this->potential(0)->data_rows(), &beta,
        this->voltage()->device_data(), this->voltage()->data_rows());

    // add harmonic voltages
    beta = 1.0f;
    for (dtype::index component = 1; component < this->model()->components_count(); ++component) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, this->voltage_calculation()->data_rows(),
            this->model()->electrodes()->drive_count(), this->voltage_calculation()->data_columns(), &alpha,
            this->voltage_calculation()->device_data(), this->voltage_calculation()->data_rows(),
            this->potential(component)->device_data(), this->potential(component)->data_rows(), &beta,
            this->voltage()->device_data(), this->voltage()->data_rows());
    }

    return this->voltage();
}

// calc jacobian
template <
    class model_type
>
void fastEIT::forward::calcJacobian(const std::shared_ptr<Matrix<dtype::real>> gamma,
    const std::shared_ptr<Matrix<dtype::real>> potential,
    const std::shared_ptr<Matrix<dtype::index>> elements,
    const std::shared_ptr<Matrix<dtype::real>> elemental_jacobian_matrix,
    dtype::size drive_count, dtype::size measurment_count, dtype::real sigma_ref,
    bool additiv, cudaStream_t stream, std::shared_ptr<Matrix<dtype::real>> jacobian) {
    // check input
    if (gamma == nullptr) {
        throw std::invalid_argument("ForwardSolver::calcJacobian: gamma == nullptr");
    }
    if (potential == nullptr) {
        throw std::invalid_argument("ForwardSolver::calcJacobian: potential == nullptr");
    }
    if (elements == nullptr) {
        throw std::invalid_argument("ForwardSolver::calcJacobian: elements == nullptr");
    }
    if (elemental_jacobian_matrix == nullptr) {
        throw std::invalid_argument("ForwardSolver::calcJacobian: elemental_jacobian_matrix == nullptr");
    }
    if (jacobian == nullptr) {
        throw std::invalid_argument("ForwardSolver::calcJacobian: jacobian == nullptr");
    }

    // dimension
    dim3 blocks(jacobian->data_rows() / matrix::block_size,
        jacobian->data_columns() / matrix::block_size);
    dim3 threads(matrix::block_size, matrix::block_size);

    // calc jacobian
    forwardKernel::calcJacobian<model_type::basis_function_type::nodes_per_element>(blocks, threads, stream,
        potential->device_data(), &potential->device_data()[drive_count * potential->data_rows()],
        elements->device_data(), elemental_jacobian_matrix->device_data(),
        gamma->device_data(), sigma_ref, jacobian->data_rows(), jacobian->data_columns(),
        potential->data_rows(), elements->rows(), drive_count, measurment_count, additiv,
        jacobian->device_data());
}

// specialisation
template void fastEIT::forward::calcJacobian<fastEIT::Model<fastEIT::basis::Linear>>(
    const std::shared_ptr<Matrix<dtype::real>>, const std::shared_ptr<Matrix<dtype::real>>,
    const std::shared_ptr<Matrix<dtype::index>>, const std::shared_ptr<Matrix<dtype::real>>,
    dtype::size, dtype::size, dtype::real, bool, cudaStream_t, std::shared_ptr<Matrix<dtype::real>>);

template class fastEIT::ForwardSolver<fastEIT::numeric::SparseConjugate,
    fastEIT::Model<fastEIT::basis::Linear>, fastEIT::source::Current>;
