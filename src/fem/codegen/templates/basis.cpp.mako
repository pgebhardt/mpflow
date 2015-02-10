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
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> nodes,
    dtype::index one)
    : mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(nodes) {
    // check one
    if (one >= pointsPerElement) {
        throw std::invalid_argument(
            "mpFlow::FEM::basis::${name}::${name}: one >= pointsPerElement");
    }

    // calc coefficients with gauss
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>
        ::Zero(pointsPerElement, pointsPerElement);
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> b = Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1>::Zero(pointsPerElement);
    b(one) = 1.0;

    // fill coefficients
    for (dtype::index node = 0; node < pointsPerElement; ++node) {
% for i in range(len(coefficients)):
        A(node, ${i}) = ${coefficients[i]};
% endfor
    }

    // calc coefficients
    this->coefficients = math::gaussElemination(A, b);
}

// evaluate basis function
${evaluate}

// integrate with basis
${integrateWithBasis}

// integrate gradient with basis
${integrateGradientWithBasis}

// integrate edge
mpFlow::dtype::real mpFlow::FEM::basis::${name}::integrateBoundaryEdge(
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> nodes, dtype::index one,
    dtype::real start, dtype::real end) {
    // calc coefficients for basis function
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> coefficients = Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1>::Zero(pointsPerEdge);
% for i in range(len(boundaryCoefficiens)):
    if (one == ${i}) {
    % for j in range(len(boundaryCoefficiens[i])):
        coefficients(${j}) = ${boundaryCoefficiens[i][j].expand()};
    % endfor
    }
% endfor
    return ${integrateBoundaryEdge};
}
