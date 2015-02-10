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

#include <cmath>
#include "mpflow/mpflow.h"

using namespace std;

// create basis class
mpFlow::FEM::basis::${name}::${name} (
    Eigen::ArrayXXf nodes,
    std::tuple<dtype::index, dtype::index> edge) :
    mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(nodes),
    nodeBasis({{ Linear(nodes, std::get<0>(edge)), Linear(nodes, std::get<1>(edge)) }}) {
    // calculate length of edge
    this->length = sqrt(math::square(nodes(std::get<1>(edge), 0) - nodes(std::get<0>(edge), 0)) +
                        math::square(nodes(std::get<1>(edge), 1) - nodes(std::get<0>(edge), 1)));
}

// evaluate basis function
${evaluate}

// integrate with basis
${integrateWithBasis}

// integrate gradient with basis
${integrateGradientWithBasis}
