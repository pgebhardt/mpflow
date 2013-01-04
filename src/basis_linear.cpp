// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cmath>
#include "../include/fasteit.h"

// create basis class
fastEIT::basis::Linear::Linear(
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
    dtype::index one)
    : fastEIT::basis::Basis<nodes_per_edge, nodes_per_element>(nodes, one) {
    // calc coefficients with gauss
    std::array<std::array<dtype::real, nodes_per_element>, nodes_per_element> A;
    std::array<dtype::real, nodes_per_element> b;
    for (int i = 0; i<nodes_per_element; ++i) {
        b[i]=0.0;
    }
    b[one] = 1.0f;

    // fill coefficients
    for (int i = 0; i< nodes_per_element; i++) {
        A[i][0] = 1.0;
        A[i][1] = std::get<0>(this->nodes()[i]);
        A[i][2] = std::get<1>(this->nodes()[i]);
    }

    // calc coefficients
    this->coefficients() = math::gaussElemination<dtype::real, nodes_per_element>(A, b);
}

// evaluate basis function
fastEIT::dtype::real fastEIT::basis::Linear::evaluate(
    std::tuple<dtype::real, dtype::real> point) {
    // calc result
    return
        this->coefficients()[0] +
        this->coefficients()[1] * std::get<0>(point) +
        this->coefficients()[2] * std::get<1>(point);
}

// integrate with basis
fastEIT::dtype::real fastEIT::basis::Linear::integrateWithBasis(
    const std::shared_ptr<Linear> other) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("basis::Linear::integrateWithBasis: other == nullptr");
    }

    // shorten variables
    dtype::real x1 = std::get<0>(this->nodes()[0]);
    dtype::real y1 = std::get<1>(this->nodes()[0]);
    dtype::real x2 = std::get<0>(this->nodes()[1]);
    dtype::real y2 = std::get<1>(this->nodes()[1]);
    dtype::real x3 = std::get<0>(this->nodes()[2]);
    dtype::real y3 = std::get<1>(this->nodes()[2]);

    dtype::real ai = this->coefficients()[0];
    dtype::real bi = this->coefficients()[1];
    dtype::real ci = this->coefficients()[2];
    dtype::real aj = other->coefficients()[0];
    dtype::real bj = other->coefficients()[1];
    dtype::real cj = other->coefficients()[2];

    // calc area
    dtype::real area = 0.5 * fabs((x2 - x1) * (y3 - y1) -
        (x3 - x1) * (y2 - y1));

    // calc integral
    dtype::real integral = 2.0f * area *
        (ai * (0.5f * aj + (1.0f / 6.0f) * bj * (x1 + x2 + x3) +
        (1.0f / 6.0f) * cj * (y1 + y2 + y3)) +
        bi * ((1.0f/ 6.0f) * aj * (x1 + x2 + x3) +
        (1.0f / 12.0f) * bj * (
            x1 * x1 + x1 * x2 + x1 * x3 + x2 * x2 + x2 * x3 + x3 * x3) +
        (1.0f/ 24.0f) * cj * (
            2.0f * x1 * y1 + x1 * y2 + x1 * y3 + x2 * y1 +
            2.0f * x2 * y2 + x2 * y3 + x3 * y1 + x3 * y2 + 2.0f * x3 * y3)) +
        ci * ((1.0f / 6.0f) * aj * (y1 + y2 + y3) +
        (1.0f / 12.0f) * cj * (
            y1 * y1 + y1 * y2 + y1 * y3 + y2 * y2 + y2 * y3 + y3 * y3) +
        (1.0f / 24.0f) * bj * (
            2.0f * x1 * y1 + x1 * y2 + x1 * y3 + x2 * y1 +
            2.0f * x2 * y2 + x2 * y3 + x3 * y1 + x3 * y2 + 2.0f * x3 * y3)));

    return integral;
}

// integrate gradient with basis
fastEIT::dtype::real fastEIT::basis::Linear::integrateGradientWithBasis(
    const std::shared_ptr<Linear> other) {
    // check input
    if (other == nullptr) {
        throw std::invalid_argument("basis::Linear::integrateGradientWithBasis: other == nullptr");
    }

    // calc area
    dtype::real area = 0.5 * fabs(
        (std::get<0>(this->nodes()[1]) - std::get<0>(this->nodes()[0])) *
        (std::get<1>(this->nodes()[2]) - std::get<1>(this->nodes()[0])) -
        (std::get<0>(this->nodes()[2]) - std::get<0>(this->nodes()[0])) *
        (std::get<1>(this->nodes()[1]) - std::get<1>(this->nodes()[0])));

    // calc integral
    return area * (this->coefficients()[1] * other->coefficients()[1] +
        this->coefficients()[2] * other->coefficients()[2]);
}

// integrate edge
fastEIT::dtype::real fastEIT::basis::Linear::integrateBoundaryEdge(
    const std::array<dtype::real, nodes_per_edge> nodes, const dtype::index one,
    const dtype::real start, const dtype::real end) {
    // integral
    dtype::real integral = 0.0;

    // first parameter set to one
    if (one == 0) {
        // check if integration interval is outside of function definition
        if ((start < nodes[1]) && (end > 0.0)) {
            // function is completely inside of integration interval
            if ((start <= 0.0) && (end >= nodes[1])) {
                integral = 0.5 * nodes[1];

            // end of integration interval is inside of function definition
            } else if ((start <= 0.0) && (end < nodes[1])) {
                integral = (end - 0.5 * math::square(end) / nodes[1]);

            // start of integration interval is inside of function definition
            } else if ((start > 0.0) && (end >= nodes[1])) {
                integral = (nodes[1] - 0.5 * math::square(nodes[1]) / nodes[1]) -
                            (start - 0.5 * math::square(start) / nodes[1]);

            // both ends of integration interval are inside of function definition
            } else if ((start > 0.0) && (end < nodes[1])) {
                integral = (end - 0.5 * math::square(end) / nodes[1]) -
                            (start - 0.5 * math::square(start) / nodes[1]);
            }
        }
    // second parameter set to one
    } else {
        // check if integration interval is outside of function definition
        if ((start < 0.0) && (end > nodes[0])) {
            // function is completely inside of integration interval
            if ((start <= nodes[0]) && (end >= 0.0)) {
                integral = -0.5f * nodes[0];

            // end of integration interval is inside of function definition
            } else if ((start <= nodes[0]) && (end < 0.0)) {
                integral = (end - 0.5 * end * end / nodes[0]) -
                           (nodes[0] - 0.5 * nodes[0] * nodes[0] / nodes[0]);

            // start of integration interval is inside of function definition
            } else if ((start > nodes[0]) && (end >= 0.0)) {
                integral = -(start - 0.5 * start * start / nodes[0]);

            // both ends of integration interval are inside of function definition
            } else if ((start > nodes[0]) && (end < 0.0)) {
                integral = (end - 0.5 * end * end / nodes[0]) -
                           (start - 0.5 * start * start / nodes[0]);
            }
        }
    }

    return integral;
}
