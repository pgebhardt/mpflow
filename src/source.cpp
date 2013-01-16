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

fastEIT::source::Current::Current(dtype::real current,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : Source(drive_pattern, measurement_pattern), current_(current) {
    // check input
    if (current <= 0.0) {
        throw std::invalid_argument("Current::Current: current <= 0.0");
    }
}

fastEIT::source::Voltage::Voltage(dtype::real voltage,
    std::shared_ptr<Matrix<dtype::real>> drive_pattern,
    std::shared_ptr<Matrix<dtype::real>> measurement_pattern)
    : Source(drive_pattern, measurement_pattern), voltage_(voltage) {
    // check input
    if (voltage <= 0.0) {
        throw std::invalid_argument("Voltage::Voltage: voltage <= 0.0");
    }
}
