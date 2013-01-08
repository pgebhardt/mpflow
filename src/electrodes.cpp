// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

// create electrodes class
template <
    class mesh_type
>
fastEIT::Electrodes<mesh_type>::Electrodes(dtype::size count,
    std::tuple<dtype::real, dtype::real> shape, const std::shared_ptr<mesh_type> mesh)
    : count_(count), shape_(shape) {
    // check input
    if (count == 0) {
        throw std::invalid_argument("Electrodes::Electrodes: count == 0");
    }
    if (std::get<0>(shape) <= 0.0) {
        throw std::invalid_argument("Electrodes::Electrodes: width <= 0.0");
    }
    if (std::get<1>(shape) <= 0.0) {
        throw std::invalid_argument("Electrodes::Electrodes: height <= 0.0");
    }
    if (mesh == nullptr) {
        throw std::invalid_argument("Electrodes::Electrodes: mesh == nullptr");
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
                angle + std::get<0>(shape) / mesh->radius()))));
    }
}

// template class specialisations
template class fastEIT::Electrodes<fastEIT::Mesh<fastEIT::basis::Linear>>;
