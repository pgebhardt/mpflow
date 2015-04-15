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
    Eigen::Ref<Eigen::ArrayXXd const> const coordinates,
    double const height)
    : coordinates(coordinates), height(height), count(coordinates.rows()) {
    // check input
    if (coordinates.rows() == 0) {
        throw std::invalid_argument("mpFlow::FEM::BoundaryDescriptor::BoundaryDescriptor: count == 0");
    }
}

std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> mpFlow::FEM::boundaryDescriptor::circularBoundary(
    unsigned const count, double const width, double const height,
    double const boundaryRadius, double const offset) {
    // check radius
    if (boundaryRadius == 0.0) {
        throw std::invalid_argument(
            "mpFlow::FEM::boundaryDescriptor::circularBoundary: boundaryRadius == 0.0");
    }

    // fill coordinates vectors
    double angle = 0.0f;
    double deltaAngle = M_PI / (double)count;
    Eigen::ArrayXXd coordinates = Eigen::ArrayXXd::Zero(count, 4);
    for (unsigned electrode = 0; electrode < count; ++electrode) {
        // calc start angle
        angle = (double)electrode * 2.0 * deltaAngle + offset / boundaryRadius;

        // calc coordinates
        Eigen::ArrayXd point(2);
        point << boundaryRadius, angle;
        coordinates.block(electrode, 0, 1, 2) = math::kartesian(point).transpose();

        point << boundaryRadius, angle + width / boundaryRadius;
        coordinates.block(electrode, 2, 1, 2) = math::kartesian(point).transpose();
    }

    return std::make_shared<BoundaryDescriptor>(coordinates, height);
}
