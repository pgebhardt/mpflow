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

std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> mpFlow::FEM::BoundaryDescriptor::circularBoundary(
    unsigned const count, double const width, double const height,
    double const boundaryRadius, double const offset, bool const clockwise) {
    // check radius
    if (boundaryRadius == 0.0) {
        throw std::invalid_argument(
            "mpFlow::FEM::boundaryDescriptor::circularBoundary: boundaryRadius == 0.0");
    }
    
    // get port start and end points
    auto const startPoints = math::circularPoints(boundaryRadius, 2.0 * M_PI * boundaryRadius / count, offset, clockwise);
    auto const endPoints = math::circularPoints(boundaryRadius, 2.0 * M_PI * boundaryRadius / count, offset + width, clockwise);
    auto const coordinates = (Eigen::ArrayXXd(count, 2 * startPoints.cols()) <<
            (clockwise ? endPoints : startPoints), (clockwise ? startPoints : endPoints)).finished();

    return std::make_shared<BoundaryDescriptor>(coordinates, height);
}

std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> mpFlow::FEM::BoundaryDescriptor::fromConfig(
    json_value const& config, std::shared_ptr<numeric::IrregularMesh const> const mesh) {
    // read out height
    auto const height = config["height"].type != json_none ? config["height"].u.dbl : 1.0;
    auto const radius = std::sqrt(mesh->nodes.square().rowwise().sum().maxCoeff());
    
    // extract descriptor coordinates from config, or create circular descriptor
    // if no coordinates are given
    if (config["coordinates"].type != json_none) {
        auto const coordinates = numeric::eigenFromJsonArray<double>(config["coordinates"]);
        
        return std::make_shared<mpFlow::FEM::BoundaryDescriptor>(coordinates, height);
    }
    else {
        auto const width = config["width"].u.dbl;
        auto const count = config["count"].u.integer;
        auto const offset = config["offset"].u.dbl;
        auto const invertDirection = config["invertDirection"].u.boolean;
        
        return mpFlow::FEM::BoundaryDescriptor::circularBoundary(count, width,
            height, radius, offset, invertDirection);
    }
}

