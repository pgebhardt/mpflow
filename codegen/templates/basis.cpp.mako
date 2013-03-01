// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cmath>
#include "fasteit/fasteit.h"

using namespace std;

// create basis class
fastEIT::basis::${name}::${name}(
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
    dtype::index one)
    : fastEIT::basis::Basis<nodes_per_edge, nodes_per_element>(nodes, one) {
    // check one
    if (one > nodes_per_element) {
        throw std::invalid_argument(
            "fastEIT::basis::${name}::${name}: one > nodes_per_element");
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
fastEIT::dtype::real fastEIT::basis::${name}::integrateBoundaryEdge(
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
