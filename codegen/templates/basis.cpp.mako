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

    // calc coefficients
% for i in range(len(coefficients)):
    if (one == ${i}) {
    % for j in range(len(coefficients[i])):
        this->coefficients()[${j}] = ${coefficients[i][j]};
    % endfor
    }
% endfor
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
    // crop integration interval to function definition
    start = std::min(std::max(nodes[0], start), nodes[nodes_per_edge - 1]);
    end = std::min(std::max(nodes[0], end), nodes[nodes_per_edge - 1]);

    // calc coefficients for basis function
    std::array<dtype::real, nodes_per_edge> coefficients;
% for i in range(len(boundaryCoefficiens)):
    if (one == ${i}) {
    % for j in range(len(boundaryCoefficiens[i])):
        coefficients[${j}] = ${boundaryCoefficiens[i][j]};
    % endfor
    }
% endfor

${integrateBoundaryEdge}
}
