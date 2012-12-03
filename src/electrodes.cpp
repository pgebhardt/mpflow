// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdexcept>
#include <assert.h>
#include <cmath>
#include <tuple>

#include "../include/dtype.hpp"
#include "../include/math.hpp"
#include "../include/electrodes.hpp"

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
    std::tuple<dtype::real, dtype::real> point;
    for (dtype::index i = 0; i < this->count(); i++) {
        // calc start angle
        angle = (dtype::real)i * 2.0f * deltaAngle;

        // calc start coordinates
        point = math::kartesian(std::make_tuple(meshRadius, angle));
        this->electrodes_start_[i * 2 + 0] = std::get<0>(point);
        this->electrodes_start_[i * 2 + 1] = std::get<1>(point);

        // calc end angle
        angle += this->width() / meshRadius;

        // calc end coordinates
        point = math::kartesian(std::make_tuple(meshRadius, angle));
        this->electrodes_end_[i * 2 + 0] = std::get<0>(point);
        this->electrodes_end_[i * 2 + 1] = std::get<1>(point);
    }
}

// delete electrodes class
fastEIT::Electrodes::~Electrodes() {
    // free electrode vectors
    delete [] this->electrodes_start_;
    delete [] this->electrodes_end_;
}
