// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "mpflow/mpflow.h"

// create electrodes class
mpFlow::EIT::Electrodes::Electrodes(dtype::size count,
    std::tuple<dtype::real, dtype::real> shape, dtype::real impedance)
    : count_(count), coordinates_(count), shape_(shape), impedance_(impedance) {
    // check input
    if (count == 0) {
        throw std::invalid_argument("mpFlow::EIT::Electrodes::Electrodes: count == 0");
    }
    if (std::get<0>(shape) <= 0.0) {
        throw std::invalid_argument("mpFlow::EIT::Electrodes::Electrodes: width <= 0.0");
    }
    if (std::get<1>(shape) <= 0.0) {
        throw std::invalid_argument("mpFlow::EIT::Electrodes::Electrodes: height <= 0.0");
    }
    if (impedance <= 0.0) {
        throw std::invalid_argument("mpFlow::EIT::Electrodes::Electrodes: impedance <= 0.0");
    }
}

// create electrodes on circular boundary
std::shared_ptr<mpFlow::EIT::Electrodes> mpFlow::EIT::electrodes::circularBoundary(
    dtype::size count, std::tuple<dtype::real, dtype::real> shape,
    dtype::real impedance, dtype::real boundary_radius) {
    // check radius
    if (boundary_radius <= 0.0) {
        throw std::invalid_argument(
            "mpFlow::EIT::electrodes::circularBoundary: boundary_radius <= 0.0");
    }

    // create electrodes
    auto electrodes = std::make_shared<Electrodes>(count, shape, impedance);

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real delta_angle = M_PI / (dtype::real)electrodes->count();
    for (dtype::index electrode = 0; electrode < electrodes->count(); ++electrode) {
        // calc start angle
        angle = (dtype::real)electrode * 2.0 * delta_angle;

        // calc coordinates
        electrodes->coordinates(electrode) = std::make_tuple(
            math::kartesian(std::make_tuple(boundary_radius, angle)),
            math::kartesian(std::make_tuple(boundary_radius,
                angle + std::get<0>(shape) / boundary_radius)));
    }

    return electrodes;
}
