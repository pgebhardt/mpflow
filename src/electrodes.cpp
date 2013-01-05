// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

// create electrodes class
fastEIT::Electrodes::Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real radius)
    : count_(count), width_(width), height_(height) {
    // check input
    if (count == 0) {
        throw std::invalid_argument("count == 0");
    }
    if (width <= 0.0f) {
        throw std::invalid_argument("width <= 0.0");
    }
    if (height <= 0.0f) {
        throw std::invalid_argument("height <= 0.0");
    }
    if (radius <= 0.0f) {
        throw std::invalid_argument("radius <= 0.0");
    }

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real delta_angle = M_PI / (dtype::real)this->count();
    for (dtype::index electrode = 0; electrode < this->count(); ++electrode) {
        // calc start angle
        angle = (dtype::real)electrode * 2.0 * delta_angle;

        // calc coordinates
        this->coordinates_.push_back(std::make_tuple(
            math::kartesian(std::make_tuple(radius, angle)),
            math::kartesian(std::make_tuple(radius, angle + this->width() / radius))));
    }
}
