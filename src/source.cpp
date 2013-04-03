// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "fasteit/fasteit.h"

template <
    class basis_function_type
>
fastEIT::source::Source<basis_function_type>::Source(std::string type, dtype::real value,
    std::shared_ptr<Mesh> mesh, std::shared_ptr<Electrodes> electrodes,
    dtype::size components_count, std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern, cublasHandle_t handle,
    cudaStream_t stream)
    : type_(type), mesh_(mesh), electrodes_(electrodes), drive_pattern_(drive_pattern),
        measurement_pattern_(measurement_pattern), value_(value),
        components_count_(components_count) {
    // check input
    if (mesh == nullptr) {
        throw std::invalid_argument("fastEIT::source::Source::Source: mesh == nullptr");
    }
    if (electrodes == nullptr) {
        throw std::invalid_argument("fastEIT::source::Source::Source: electrodes == nullptr");
    }
    if (drive_pattern == nullptr) {
        throw std::invalid_argument("fastEIT::source::Source::Source: drive_pattern == nullptr");
    }
    if (measurement_pattern == nullptr) {
        throw std::invalid_argument(
            "fastEIT::source::Source::Source: measurement_pattern == nullptr");
    }
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::source::Source::Source: handle == nullptr");
    }

    // create matrices
    this->pattern_ = std::make_shared<Matrix<dtype::real>>(
        this->electrodes()->count(), this->drive_count() + this->measurement_count(), stream);
    this->elemental_pattern_[0] = std::make_shared<Matrix<dtype::real>>(
        this->electrodes()->count(), this->drive_count() + this->measurement_count(), stream);
    this->elemental_pattern_[1] = std::make_shared<Matrix<dtype::real>>(
        this->electrodes()->count(), this->drive_count() + this->measurement_count(), stream);
    this->d_matrix_ = std::make_shared<Matrix<dtype::real>>(this->electrodes()->count(),
        this->electrodes()->count(), stream);
    this->w_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        this->electrodes()->count(), stream);
    this->x_matrix_ = std::make_shared<Matrix<dtype::real>>(this->electrodes()->count(),
        this->mesh()->nodes()->rows(), stream);
    this->z_matrix_ = std::make_shared<Matrix<dtype::real>>(this->mesh()->nodes()->rows(),
        this->mesh()->nodes()->rows(), stream);

    // excitation matrices
    for (dtype::index component = 0; component < this->components_count(); ++component) {
        this->excitation_.push_back(std::make_shared<Matrix<dtype::real>>(
            this->mesh()->nodes()->rows() + this->electrodes()->count(),
            this->drive_count() + this->measurement_count(), stream));
    }

    // fill pattern matrix with drive pattern
    for (dtype::index column = 0; column < this->drive_count(); ++column) {
        for (dtype::index row = 0; row < this->electrodes()->count(); ++row) {
            (*this->elemental_pattern(0))(row, column) =
                (*this->drive_pattern())(row, column);
        }
    }

    // fill pattern matrix with measurment pattern and turn sign of measurment
    // for correct current pattern
    for (dtype::index column = 0; column < this->measurement_count(); ++column) {
        for (dtype::index row = 0; row < this->electrodes()->count(); ++row) {
            (*this->elemental_pattern(1))(row, column + this->drive_count()) =
                (*this->measurement_pattern())(row, column);
        }
    }
    this->elemental_pattern(0)->copyToDevice(stream);
    this->elemental_pattern(1)->copyToDevice(stream);
}

// current source
template <
    class basis_function_type
>
fastEIT::source::Current<basis_function_type>::Current(
    dtype::real current, std::shared_ptr<Mesh> mesh,
    std::shared_ptr<Electrodes> electrodes, dtype::size components_count,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
    cublasHandle_t handle, cudaStream_t stream)
    : Source<basis_function_type>("current", current, mesh, electrodes, components_count,
        drive_pattern, measurement_pattern, handle, stream) {
    // init complete electrode model
    this->initCEM(handle, stream);

    // update excitation
    this->updateExcitation(handle, stream);
}

// init complete electrode model matrices
template <
    class basis_function_type
>
void fastEIT::source::Current<basis_function_type>::initCEM(cublasHandle_t handle,
    cudaStream_t stream) {
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::source::Current::initCEM: handle == nullptr");
    }

    // needed arrays
    std::vector<std::tuple<dtype::index, std::tuple<dtype::real, dtype::real>>> nodes;
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
                std::get<0>(this->electrodes()->coordinates(electrode)), parameter_offset);
            integration_end = math::circleParameter(
                std::get<1>(this->electrodes()->coordinates(electrode)), parameter_offset);

            // intgrate if integration_start is left of integration_end
            if (integration_start < integration_end) {
                // calc z matrix element
                for (dtype::index i = 0; i < basis_function_type::nodes_per_edge; ++i) {
                    for (dtype::index j = 0; j < basis_function_type::nodes_per_edge; ++j) {
                        // calc z matrix element
                        (*this->z_matrix())(std::get<0>(nodes[i]), std::get<0>(nodes[j])) +=
                            basis_function_type::integrateBoundaryEdgeWithOther(
                                node_parameter, i, j, integration_start, integration_end) /
                            this->electrodes()->impedance();
                    }

                    // calc w and x matrix elements
                    (*this->w_matrix())(std::get<0>(nodes[i]), electrode) -=
                        basis_function_type::integrateBoundaryEdge(
                            node_parameter, i, integration_start, integration_end) /
                        this->electrodes()->impedance();
                    (*this->x_matrix())(electrode, std::get<0>(nodes[i])) -=
                        basis_function_type::integrateBoundaryEdge(
                            node_parameter, i, integration_start, integration_end) /
                        this->electrodes()->impedance();
                }
            }
        }
    }
    this->z_matrix()->copyToDevice(stream);

    // init d matrix
    for (dtype::index electrode = 0; electrode < this->electrodes()->count(); ++electrode) {
        (*this->d_matrix())(electrode, electrode) = std::get<0>(this->electrodes()->shape()) /
            this->electrodes()->impedance();
    }
}

// update excitation
template <
    class basis_function_type
>
void fastEIT::source::Current<basis_function_type>::updateExcitation(
    cublasHandle_t handle, cudaStream_t stream) {
    if (handle == nullptr) {
        throw std::invalid_argument("fastEIT::source::Current::updateExcitation: handle == nullptr");
    }

    // calc pattern
    this->pattern()->copy(this->elemental_pattern(0), stream);
    this->pattern()->scalarMultiply(this->value(), stream);
    this->pattern()->add(this->elemental_pattern(1), stream);
    this->pattern()->copyToHost(stream);
    cudaStreamSynchronize(stream);

    // update excitation
    // calc excitation components
    for (dtype::index component = 0; component < this->components_count(); ++component) {
        // set excitation
        for (dtype::index row = 0; row < this->pattern()->rows(); ++row)
        for (dtype::index column = 0; column < this->pattern()->columns(); ++column) {
            (*this->excitation(component))(
                this->mesh()->nodes()->rows() + row, column) =
                (*this->pattern())(row, column);
        }
        this->excitation(component)->copyToDevice(stream);

        // fourier transform pattern
        if (component == 0) {
            // calc ground mode
            this->excitation(component)->scalarMultiply(1.0f / this->mesh()->height(), stream);
        } else {
            this->excitation(component)->scalarMultiply(2.0f * sin(
                component * M_PI * std::get<1>(this->electrodes()->shape()) / this->mesh()->height()) /
                (component * M_PI * std::get<1>(this->electrodes()->shape())), stream);
        }
    }
}

// Voltage source
template <
    class basis_function_type
>
fastEIT::source::Voltage<basis_function_type>::Voltage(
    dtype::real voltage, std::shared_ptr<Mesh> mesh,
    std::shared_ptr<Electrodes> electrodes, dtype::size components_count,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
    cublasHandle_t handle, cudaStream_t stream)
    : Source<basis_function_type>("voltage", voltage, mesh, electrodes, components_count,
        drive_pattern, measurement_pattern, handle, stream) {
    // init complete electrode model
    this->initCEM(handle, stream);

    // update excitation
    this->updateExcitation(handle, stream);
}

// init complete electrode model matrices
template <
    class basis_function_type
>
void fastEIT::source::Voltage<basis_function_type>::initCEM(cublasHandle_t, cudaStream_t) {
}

// update excitation
template <
    class model_type
>
void fastEIT::source::Voltage<model_type>::updateExcitation(
    cublasHandle_t, cudaStream_t) {
}

// specialisation
template class fastEIT::source::Source<fastEIT::basis::Linear>;
template class fastEIT::source::Current<fastEIT::basis::Linear>;
template class fastEIT::source::Voltage<fastEIT::basis::Linear>;
template class fastEIT::source::Source<fastEIT::basis::Quadratic>;
template class fastEIT::source::Current<fastEIT::basis::Quadratic>;
template class fastEIT::source::Voltage<fastEIT::basis::Quadratic>;
