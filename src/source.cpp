// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

template <
    class model_type
>
fastEIT::source::Source<model_type>::Source(std::string type, dtype::real value,
    std::shared_ptr<model_type> model, std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern, cublasHandle_t handle,
    cudaStream_t stream)
    : type_(type), model_(model), drive_pattern_(drive_pattern),
        measurement_pattern_(measurement_pattern), value_(value) {
    // check input
    if (drive_pattern == nullptr) {
        throw std::invalid_argument("Source::Source: drive_pattern == nullptr");
    }
    if (measurement_pattern == nullptr) {
        throw std::invalid_argument("Source::Source: measurement_pattern == nullptr");
    }
    if (model == nullptr) {
        throw std::invalid_argument("Source::Source: model == nullptr");
    }
    if (handle == NULL) {
        throw std::invalid_argument("Source::Source: handle == NULL");
    }

    // create matrices
    this->pattern_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->electrodes()->count(), this->drive_count() + this->measurement_count(), stream);
    this->elemental_pattern_[0] = std::make_shared<Matrix<dtype::real>>(
        this->model()->electrodes()->count(), this->drive_count() + this->measurement_count(), stream);
    this->elemental_pattern_[1] = std::make_shared<Matrix<dtype::real>>(
        this->model()->electrodes()->count(), this->drive_count() + this->measurement_count(), stream);
    this->excitation_matrix_ = std::make_shared<Matrix<dtype::real>>(
        this->model()->mesh()->nodes()->rows(), this->model()->electrodes()->count(), stream);

    // excitation matrices
    for (dtype::index component = 0;
        component < this->model()->components_count() + 1;
        ++component) {
        this->excitation_.push_back(std::make_shared<Matrix<dtype::real>>(
            this->model()->mesh()->nodes()->rows(),
            this->drive_count() + this->measurement_count(), stream));
    }
}

// init excitation matrix and pattern
template <
    class model_type
>
void fastEIT::source::Source<model_type>::initExcitation(cublasHandle_t handle,
    cudaStream_t stream) {
    if (handle == NULL) {
        throw std::invalid_argument("Current::initExcitation: handle == NULL");
    }

    // needed arrays
    std::array<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>,
        model_type::basis_function_type::nodes_per_edge> nodes;
    std::array<dtype::real, model_type::basis_function_type::nodes_per_edge> node_parameter;
    dtype::real integration_start, integration_end;

    // calc excitation matrix
    for (dtype::index boundary_element = 0;
        boundary_element < this->model()->mesh()->boundary()->rows();
        ++boundary_element) {
        // get boundary nodes
        nodes = this->model()->mesh()->boundaryNodes(boundary_element);

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
        for (dtype::size i = 0; i < model_type::basis_function_type::nodes_per_edge; ++i) {
            node_parameter[i] = math::circleParameter(std::get<1>(nodes[i]),
                parameter_offset);
        }

        for (dtype::index electrode = 0; electrode < this->model()->electrodes()->count(); ++electrode) {
            // calc integration interval centered to node 0
            integration_start = math::circleParameter(
                std::get<0>(this->model()->electrodes()->coordinates(electrode)),
                parameter_offset);
            integration_end = math::circleParameter(
                std::get<1>(this->model()->electrodes()->coordinates(electrode)),
                parameter_offset);

            // intgrate if integration_start is left of integration_end
            if (integration_start < integration_end) {
                // calc element
                for (dtype::index node = 0;
                    node < model_type::basis_function_type::nodes_per_edge;
                    ++node) {
                    (*this->excitation_matrix())(std::get<0>(nodes[node]), electrode) +=
                        model_type::basis_function_type::integrateBoundaryEdge(
                            node_parameter, node, integration_start, integration_end) /
                        std::get<0>(this->model()->electrodes()->shape());
                }
            }
        }
    }
    this->excitation_matrix()->copyToDevice(stream);

    // fill pattern matrix with drive pattern
    for (dtype::index column = 0; column < this->drive_count(); ++column) {
        for (dtype::index row = 0; row < this->model()->electrodes()->count(); ++row) {
            (*this->elemental_pattern(0))(row, column) =
                (*this->drive_pattern())(row, column);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::index column = 0; column < this->measurement_count(); ++column) {
        for (dtype::index row = 0; row < this->model()->electrodes()->count(); ++row) {
            (*this->elemental_pattern(1))(row, column + this->drive_count()) =
                (*this->measurement_pattern())(row, column);
        }
    }
    this->elemental_pattern(0)->copyToDevice(stream);
    this->elemental_pattern(1)->copyToDevice(stream);

    // update excitation
    this->updateExcitation(handle, stream);
}

// current source
template <
    class model_type
>
fastEIT::source::Current<model_type>::Current(
    dtype::real current, std::shared_ptr<model_type> model,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
    cublasHandle_t handle, cudaStream_t stream)
    : Source<model_type>("current", current, model, drive_pattern,
        measurement_pattern, handle, stream) {

    // init excitation matrix
    this->initExcitation(handle, stream);
}

// update excitation
template <
    class model_type
>
void fastEIT::source::Current<model_type>::updateExcitation(
    cublasHandle_t handle, cudaStream_t stream) {
    if (handle == NULL) {
        throw std::invalid_argument("Current::updateExcitation: handle == NULL");
    }

    // calc pattern
    this->pattern()->copy(this->elemental_pattern(0), stream);
    this->pattern()->scalarMultiply(this->value(), stream);
    this->pattern()->add(this->elemental_pattern(1), stream);

    // update excitation
    // calc excitation components
    for (dtype::index component = 0; component < this->model()->components_count() + 1; ++component) {
        // set excitation
        this->excitation(component)->multiply(this->excitation_matrix(),
            this->pattern(), handle, stream);

        // fourier transform pattern
        if (component == 0) {
            // calc ground mode
            this->excitation(component)->scalarMultiply(1.0f / this->model()->mesh()->height(), stream);
        } else {
            this->excitation(component)->scalarMultiply(2.0f * sin(
                component * M_PI * std::get<1>(this->model()->electrodes()->shape()) / this->model()->mesh()->height()) /
                (component * M_PI * std::get<1>(this->model()->electrodes()->shape())), stream);
        }
    }
}

template <
    class model_type
>
fastEIT::source::Voltage<model_type>::Voltage(
    dtype::real voltage, std::shared_ptr<model_type> model,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
    cublasHandle_t handle, cudaStream_t stream)
    : Source<model_type>("voltage", voltage, model, drive_pattern,
        measurement_pattern, handle, stream) {

    // init excitation matrix
    this->initExcitation(handle, stream);
}

// update excitation
template <
    class model_type
>
void fastEIT::source::Voltage<model_type>::updateExcitation(
    cublasHandle_t, cudaStream_t) {
}
// specialisation
template class fastEIT::source::Current<fastEIT::Model<fastEIT::basis::Linear>>;
template class fastEIT::source::Voltage<fastEIT::Model<fastEIT::basis::Linear>>;
