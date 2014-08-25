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

mpFlow::FEM::BoundaryDescriptor::BoundaryDescriptor(
    const std::vector<std::tuple<dtype::real, dtype::real>>& shapes)
    : count(shapes.size()), coordinates(shapes.size()), shapes(shapes) {
    // check input
    if (this->count == 0) {
        throw std::invalid_argument("mpFlow::FEM::BoundaryDescriptor::BoundaryDescriptor: count == 0");
    }
}

mpFlow::FEM::BoundaryDescriptor::BoundaryDescriptor(dtype::size count,
    std::tuple<dtype::real, dtype::real> shape)
    : BoundaryDescriptor(std::vector<std::tuple<dtype::real, dtype::real>>(count, shape)) {
}

std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> mpFlow::FEM::boundaryDescriptor::circularBoundary(
    dtype::size count, std::tuple<dtype::real, dtype::real> shape,
    dtype::real boundaryRadius, dtype::real offset) {
    // check radius
    if (boundaryRadius == 0.0) {
        throw std::invalid_argument(
            "mpFlow::FEM::boundaryDescriptor::circularBoundary: boundaryRadius == 0.0");
    }

    auto descriptor = std::make_shared<BoundaryDescriptor>(count, shape);

    // fill coordinates vectors
    dtype::real angle = 0.0f;
    dtype::real deltaAngle = M_PI / (dtype::real)descriptor->count;
    for (dtype::index electrode = 0; electrode < descriptor->count; ++electrode) {
        // calc start angle
        angle = (dtype::real)electrode * 2.0 * deltaAngle + offset / boundaryRadius;

        // calc coordinates
        descriptor->coordinates[electrode] = std::make_tuple(
            math::kartesian(std::make_tuple(boundaryRadius, angle)),
            math::kartesian(std::make_tuple(boundaryRadius,
                angle + std::get<0>(shape) / boundaryRadius)));
    }

    return descriptor;
}
