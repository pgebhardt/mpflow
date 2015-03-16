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
    Eigen::Ref<const Eigen::ArrayXXd> points,
    Eigen::Ref<const Eigen::ArrayXi> edge) :
    mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(points),
    nodeBasis({{ Linear(points, edge(0)), Linear(points, edge(1)) }}) {
    // calculate length of edge
    this->length = sqrt(math::square(points(edge(1), 0) - points(edge(0), 0)) + math::square(points(edge(1), 1) - points(edge(0), 1)));
}

// integrate with basis
${integrateWithBasis}

// integrate gradient with basis
${integrateGradientWithBasis}
