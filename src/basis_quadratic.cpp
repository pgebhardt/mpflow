// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cmath>
#include "../include/fasteit.h"

fastEIT::basis::Quadratic::Quadratic(
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
    dtype::index one)
    : fastEIT::basis::Basis<nodes_per_edge, nodes_per_element>(nodes, one) {
    // TODO
}

// evaluate basis function
fastEIT::dtype::real fastEIT::basis::Quadratic::evaluate(
    std::tuple<dtype::real, dtype::real> point) {
    // calc result
    // TODO
    return 0.0;
}

// integrate with basis
fastEIT::dtype::real fastEIT::basis::Quadratic::integrateWithBasis(
    const std::shared_ptr<Quadratic> other) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("basis::Quadratic::integrateWithBasis: other == nullptr");
    }

    // TODO
    return 0.0;
}

// integrate gradient with basis
fastEIT::dtype::real fastEIT::basis::Quadratic::integrateGradientWithBasis(
    const std::shared_ptr<Quadratic> other) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("basis::Quadratic::integrateGradientWithBasis: other == nullptr");
    }

    // TODO
    return 0.0;
}

// integrate edge
fastEIT::dtype::real fastEIT::basis::Quadratic::integrateBoundaryEdge(
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_edge> nodes,
    const std::tuple<dtype::real, dtype::real> start,
    const std::tuple<dtype::real, dtype::real> end) {
    // TODO
    return 0.0;
}
