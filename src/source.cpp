// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

fastEIT::source::Current::Current(dtype::real current,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : Source(drive_pattern->columns(), measurement_pattern->columns()),
        current_(current), drive_pattern_(drive_pattern),
        measurement_pattern_(measurement_pattern) {
    // check input
    if (current <= 0.0) {
        throw std::invalid_argument("Current::Current: current <= 0.0");
    }
}
