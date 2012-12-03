// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// create electrodes class
fastEIT::Electrodes::Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real meshRadius)
    : count_(count), electrodes_start_(NULL), electrodes_end_(NULL), width_(width), height_(height) {
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

    // create electrode vectors
    this->electrodes_start_ = new dtype::real[this->count() * 2];
    this->electrodes_end_ = new dtype::real[this->count() * 2];

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real deltaAngle = M_PI / (dtype::real)this->count();
    for (dtype::index i = 0; i < this->count(); i++) {
        // calc start angle
        angle = (dtype::real)i * 2.0f * deltaAngle;

        // calc start coordinates
        math::kartesian(this->electrodes_start_[i * 2 + 0], this->electrodes_start_[i * 2 + 1],
            meshRadius, angle);

        // calc end angle
        angle += this->width() / meshRadius;

        // calc end coordinates
        math::kartesian(this->electrodes_end_[i * 2 + 0], this->electrodes_end_[i * 2 + 1],
            meshRadius, angle);
    }
}

// delete electrodes class
fastEIT::Electrodes::~Electrodes() {
    // free electrode vectors
    delete [] this->electrodes_start_;
    delete [] this->electrodes_end_;
}
