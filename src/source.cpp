// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

fastEIT::source::Source::Source(std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : drive_pattern_(drive_pattern), measurement_pattern_(measurement_pattern) {
    // check input
    if (drive_pattern == nullptr) {
        throw std::invalid_argument("Source::Source: drive_pattern == nullptr");
    }
    if (measurement_pattern == nullptr) {
        throw std::invalid_argument("Source::Source: measurement_pattern == nullptr");
    }
}
