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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#include "json.h"
#include "mpflow/mpflow.h"

mpFlow::FEM::Ports::Ports(Eigen::Ref<Eigen::ArrayXXi const> const edges)
    : edges(edges), count(edges.cols()) {
    // check input
    if (edges.rows() == 0) {
        throw std::invalid_argument("mpFlow::FEM::Ports::Ports: edges == 0");
    }
}

 std::shared_ptr<mpFlow::FEM::Ports> mpFlow::FEM::Ports::circularBoundary(
    unsigned const count, double const width, std::shared_ptr<numeric::IrregularMesh const> const mesh,
    double const offset, bool const clockwise) {
    double radius = 0.0;
    for (int element = 0; element < mesh->elements.rows(); ++element) {
        radius = std::max(radius, std::sqrt(mesh->nodes.row(mesh->elements(element, 0)).square().sum()));
        radius = std::max(radius, std::sqrt(mesh->nodes.row(mesh->elements(element, 1)).square().sum()));
        radius = std::max(radius, std::sqrt(mesh->nodes.row(mesh->elements(element, 2)).square().sum()));
    }
    
    // find edges located inside port interval on a circular boundary
    auto const portStart = math::circularPoints(radius, 2.0 * M_PI * radius / count,
        offset + (clockwise ? width : 0.0), clockwise);
    auto const portEnd = math::circularPoints(radius, 2.0 * M_PI * radius / count,
        offset + (clockwise ? 0.0 : width), clockwise);

    std::vector<std::vector<int>> portEdgesVector(count);
    unsigned maxPortEdges = 0;
    for (unsigned port = 0; port < count; ++port) {
        for (unsigned edge = 0; edge < mesh->boundary.rows(); ++edge) {
            // calculate interval parameter
            auto const nodes = mesh->edgeNodes(mesh->boundary(edge));
            auto const nodeParameter = sqrt((nodes.rowwise() - nodes.row(0)).square().rowwise().sum()).eval();

            auto const parameterOffset = math::circleParameter(nodes.row(0).transpose(), 0.0);
            auto const intervalStart = math::circleParameter(portStart.row(port).transpose(), parameterOffset);
            auto const intervalEnd = math::circleParameter(portEnd.row(port).transpose(), parameterOffset);
            
            // check if edge lies within port interval
            if ((intervalStart < intervalEnd) && (nodeParameter(0) - intervalStart >= -1e-9) &&
                (nodeParameter(nodeParameter.rows() - 1) - intervalEnd <= 1e-9)) {
                portEdgesVector[port].push_back(mesh->boundary(edge));
                maxPortEdges = std::max(maxPortEdges, (unsigned)portEdgesVector[port].size());
            }
        }
    }
    
    // convert stl vector to eigen array
    Eigen::ArrayXXi portEdges = Eigen::ArrayXXi::Ones(maxPortEdges, count) * constants::invalidIndex;
    for (unsigned port = 0; port < count; ++port)
    for (unsigned edge = 0; edge < portEdgesVector[port].size(); ++edge) {
        portEdges(edge, port) = portEdgesVector[port][edge];
    }
    
    return std::make_shared<Ports>(portEdges);
}

std::shared_ptr<mpFlow::FEM::Ports> mpFlow::FEM::Ports::fromConfig(
    json_value const& config, std::shared_ptr<numeric::IrregularMesh const> const mesh,
    cudaStream_t const stream, std::string const path) {
    // check whether a path to a file is given, or a complete config
    if (config.type == json_string) {
        // read out port edges from file
        auto const ports = numeric::Matrix<int>::loadtxt(str::format("%s/%s")
            (path, std::string(config)), stream)->toEigen();
            
        return std::make_shared<Ports>(ports);      
    }
    else if (config["edges"].type == json_string) {
        // read out port edges from file
        auto const ports = numeric::Matrix<int>::loadtxt(str::format("%s/%s")
            (path, std::string(config["edges"])), stream)->toEigen();

        return std::make_shared<Ports>(ports);        
    }
    else {
        // read out basic config
        auto const width = config["width"].u.dbl;
        auto const count = config["count"].u.integer;
        auto const offset = config["offset"].u.dbl;
        auto const clockwise = config["clockwise"].u.boolean;
        
        return circularBoundary(count, width, mesh, offset, clockwise);
    }
}

