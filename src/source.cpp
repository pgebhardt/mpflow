// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.h"

fastEIT::source::Source::Source(dtype::real value,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : drive_pattern_(drive_pattern), measurement_pattern_(measurement_pattern), value_(value) {
    // check input
    if (drive_pattern == nullptr) {
        throw std::invalid_argument("Source::Source: drive_pattern == nullptr");
    }
    if (measurement_pattern == nullptr) {
        throw std::invalid_argument("Source::Source: measurement_pattern == nullptr");
    }
}

fastEIT::source::Current::Current(dtype::real current,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : Source(current, drive_pattern, measurement_pattern) {
}

fastEIT::source::Voltage::Voltage(dtype::real voltage,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : Source(voltage, drive_pattern, measurement_pattern) {
}
