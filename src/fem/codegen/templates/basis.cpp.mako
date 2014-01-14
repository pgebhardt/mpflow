// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cmath>
#include "mpflow/mpflow.h"

using namespace std;

// create basis class
mpFlow::FEM::basis::${name}::${name}(
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
    dtype::index one)
    : mpFlow::FEM::basis::Basis<nodes_per_edge, nodes_per_element>(nodes, one) {
    // check one
    if (one >= nodes_per_element) {
        throw std::invalid_argument(
            "mpFlow::FEM::basis::${name}::${name}: one >= nodes_per_element");
    }

    // calc coefficients with gauss
    std::array<std::array<dtype::real, nodes_per_element>, nodes_per_element> A;
    std::array<dtype::real, nodes_per_element> b;
    for (dtype::index node = 0; node < nodes_per_element; ++node) {
        b[node] = 0.0;
    }
    b[one] = 1.0;

    // fill coefficients
    for (dtype::index node = 0; node < nodes_per_element; ++node) {
% for i in range(len(coefficients)):
        A[node][${i}] = ${coefficients[i]};
% endfor
    }

    // calc coefficients
    this->coefficients() = math::gaussElemination<dtype::real, nodes_per_element>(A, b);
}

// evaluate basis function
${evaluate}

// integrate with basis
${integrateWithBasis}

// integrate gradient with basis
${integrateGradientWithBasis}

// integrate edge
mpFlow::dtype::real mpFlow::FEM::basis::${name}::integrateBoundaryEdge(
    std::array<dtype::real, nodes_per_edge> nodes, dtype::index one,
    dtype::real start, dtype::real end) {
    // calc coefficients for basis function
    std::array<dtype::real, nodes_per_edge> coefficients;
% for i in range(len(boundaryCoefficiens)):
    if (one == ${i}) {
    % for j in range(len(boundaryCoefficiens[i])):
        coefficients[${j}] = ${boundaryCoefficiens[i][j].expand()};
    % endfor
    }
% endfor
    return ${integrateBoundaryEdge};
}

// integrate edge with other
mpFlow::dtype::real mpFlow::FEM::basis::${name}::integrateBoundaryEdgeWithOther(
    std::array<dtype::real, nodes_per_edge> nodes, dtype::index self,
    dtype::index other, dtype::real start, dtype::real end) {
    // calc coefficients for basis function
    std::array<dtype::real, nodes_per_edge> self_coefficients, other_coefficients;
% for i in range(len(boundaryCoefficiens)):
    if (self == ${i}) {
    % for j in range(len(boundaryCoefficiens[i])):
        self_coefficients[${j}] = ${boundaryCoefficiens[i][j].expand()};
    % endfor
    }
    if (other == ${i}) {
    % for j in range(len(boundaryCoefficiens[i])):
        other_coefficients[${j}] = ${boundaryCoefficiens[i][j].expand()};
    % endfor
    }
% endfor
    return ${integrateBoundaryEdgeWithOther};
}
