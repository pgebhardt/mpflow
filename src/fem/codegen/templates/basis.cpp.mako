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
mpFlow::FEM::basis::${name}::${name}(
    std::array<std::tuple<double, double>, pointsPerElement> nodes,
    dtype::index one)
    : mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(nodes) {
    // check one
    if (one >= pointsPerElement) {
        throw std::invalid_argument(
            "mpFlow::FEM::basis::${name}::${name}: one >= pointsPerElement");
    }

    // calc coefficients with gauss
    std::array<std::array<double, pointsPerElement>, pointsPerElement> A;
    std::array<double, pointsPerElement> b;
    for (dtype::index node = 0; node < pointsPerElement; ++node) {
        b[node] = 0.0;
    }
    b[one] = 1.0;

    // fill coefficients
    for (dtype::index node = 0; node < pointsPerElement; ++node) {
% for i in range(len(coefficients)):
        A[node][${i}] = ${coefficients[i]};
% endfor
    }

    // calc coefficients
    this->coefficients = math::gaussElemination<double, pointsPerElement>(A, b);
}

// evaluate basis function
${evaluate}

// integrate with basis
${integrateWithBasis}

// integrate gradient with basis
${integrateGradientWithBasis}

// integrate edge
double mpFlow::FEM::basis::${name}::integrateBoundaryEdge(
    std::array<double, pointsPerEdge> nodes, dtype::index one,
    double start, double end) {
    // calc coefficients for basis function
    std::array<double, pointsPerEdge> coefficients;
% for i in range(len(boundaryCoefficiens) - 1):
    if (one == ${i}) {
    % for j in range(len(boundaryCoefficiens[i])):
        coefficients[${j}] = ${boundaryCoefficiens[i][j].expand()};
    % endfor
    }
    else
% endfor
    {
    % for j in range(len(boundaryCoefficiens[i])):
        coefficients[${j}] = ${boundaryCoefficiens[i][j].expand()};
    % endfor
    }
    return ${integrateBoundaryEdge};
}
