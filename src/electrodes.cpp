// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

// create electrodes class
template <
    class MeshType
>
fastEIT::Electrodes<MeshType>::Electrodes(dtype::size count, dtype::real width, dtype::real height,
            const std::shared_ptr<MeshType> mesh,
            std::shared_ptr<Matrix<dtype::real>> drive_pattern,
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : count_(count), width_(width), height_(height), drive_pattern_(drive_pattern),
        measurement_pattern_(measurement_pattern) {
    // check input
    if (count == 0) {
        throw std::invalid_argument("Electrodes::Electrodes: count == 0");
    }
    if (width <= 0.0) {
        throw std::invalid_argument("Electrodes::Electrodes: width <= 0.0");
    }
    if (height <= 0.0) {
        throw std::invalid_argument("Electrodes::Electrodes: height <= 0.0");
    }
    if (mesh == nullptr) {
        throw std::invalid_argument("Electrodes::Electrodes: mesh == nullptr");
    }
    if (drive_pattern == nullptr) {
        throw std::invalid_argument("Electrodes::Electrodes: drive_pattern == nullptr");
    }
    if (measurement_pattern == nullptr) {
        throw std::invalid_argument("Electrodes::Electrodes: measurement_pattern == nullptr");
    }

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real delta_angle = M_PI / (dtype::real)this->count();
    for (dtype::index electrode = 0; electrode < this->count(); ++electrode) {
        // calc start angle
        angle = (dtype::real)electrode * 2.0 * delta_angle;

        // calc coordinates
        this->coordinates_.push_back(std::make_tuple(
            math::kartesian(std::make_tuple(mesh->radius(), angle)),
            math::kartesian(std::make_tuple(mesh->radius(),
                angle + this->width() / mesh->radius()))));
    }
}

// template class specialisations
template class fastEIT::Electrodes<fastEIT::Mesh<fastEIT::basis::Linear>>;
