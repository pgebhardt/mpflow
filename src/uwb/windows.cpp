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

// create gc class
mpFlow::UWB::Windows::Windows(dtype::size count,
    std::tuple<dtype::real, dtype::real> shape)
    : _count(count), _coordinates(count), _shape(shape) {
    // check input
    if (count == 0) {
        throw std::invalid_argument("mpFlow::UWB::Windows::Windows: count == 0");
    }
    if (std::get<0>(shape) <= 0.0) {
        throw std::invalid_argument("mpFlow::UWB::Windows::Windows: width <= 0.0");
    }
    if (std::get<1>(shape) <= 0.0) {
        throw std::invalid_argument("mpFlow::UWB::Windows::Windows: height <= 0.0");
    }
}

// create gc on circular boundary
std::shared_ptr<mpFlow::UWB::Windows> mpFlow::UWB::windows::circularBoundary(
    dtype::size count, std::tuple<dtype::real, dtype::real> shape,
    dtype::real boundary_radius) {
    // check radius
    if (boundary_radius <= 0.0) {
        throw std::invalid_argument(
            "mpFlow::UWB::windows::circularBoundary: boundary_radius <= 0.0");
    }

    // create gc
    auto windows = std::make_shared<Windows>(count, shape);

    // fill electrode vectors
    dtype::real angle = 0.0f;
    dtype::real delta_angle = M_PI / (dtype::real)windows->count();
    for (dtype::index electrode = 0; electrode < windows->count(); ++electrode) {
        // calc start angle
        angle = (dtype::real)electrode * 2.0 * delta_angle;

        // calc coordinates
        windows->coordinates(electrode) = std::make_tuple(
            math::kartesian(std::make_tuple(boundary_radius, angle)),
            math::kartesian(std::make_tuple(boundary_radius,
                angle + std::get<0>(shape) / boundary_radius)));
    }

    return windows;
}
