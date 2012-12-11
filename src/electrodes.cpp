// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

// create electrodes class
fastEIT::Electrodes::Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real meshRadius)
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
    if (meshRadius <= 0.0f) {
        throw std::invalid_argument("meshRadius <= 0.0");
    }

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real deltaAngle = M_PI / (dtype::real)this->count();
    for (dtype::index i = 0; i < this->count(); i++) {
        // calc start angle
        angle = (dtype::real)i * 2.0f * deltaAngle;

        // calc start coordinates
        this->electrodes_start_.push_back(math::kartesian(std::make_tuple(meshRadius, angle)));

        // calc end angle
        angle += this->width() / meshRadius;

        // calc end coordinates
        this->electrodes_end_.push_back(math::kartesian(std::make_tuple(meshRadius, angle)));
    }
}
