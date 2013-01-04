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
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_edge> nodes,
    const std::tuple<dtype::real, dtype::real> start,
    const std::tuple<dtype::real, dtype::real> end) {
    // integral
    dtype::real integral = 0.0f;

    // calc node parameter
    dtype::real* nodeParameter = new dtype::real[nodes_per_edge];
    nodeParameter[0] = 0.0f;
    for (dtype::size i = 0; i < nodes_per_edge; i++) {
        nodeParameter[i] = math::circleParameter(nodes[i], nodeParameter[0]);
    }

    // calc integration boundary parameter
    dtype::real boundaryParameter[2];
    boundaryParameter[0] = math::circleParameter(start, nodeParameter[0]);
    boundaryParameter[1] = math::circleParameter(end, nodeParameter[0]);

    // integrate left triangle
    if (nodeParameter[1] < 0.0f) {
        if ((boundaryParameter[0] < 0.0f) && (boundaryParameter[1] > nodeParameter[1])) {
            if ((boundaryParameter[1] >= 0.0f) && (boundaryParameter[0] <= nodeParameter[1])) {
                integral = -0.5f * nodeParameter[1];

            } else if ((boundaryParameter[1] >= 0.0f) && (boundaryParameter[0] > nodeParameter[1])) {
                integral = -(boundaryParameter[0] - 0.5 * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);

            } else if ((boundaryParameter[1] < 0.0f) && (boundaryParameter[0] <= nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5 * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]) -
                           (nodeParameter[1] - 0.5 * nodeParameter[1] * nodeParameter[1] / nodeParameter[1]);

            } else if ((boundaryParameter[1] < 0.0f) && (boundaryParameter[0] > nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5 * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]) -
                           (boundaryParameter[0] - 0.5 * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);
            }
        }
    } else {
        // integrate right triangle
        if ((boundaryParameter[1] > 0.0f) && (nodeParameter[1] > boundaryParameter[0])) {
            if ((boundaryParameter[0] <= 0.0f) && (boundaryParameter[1] >= nodeParameter[1])) {
                integral = 0.5f * nodeParameter[1];

            } else if ((boundaryParameter[0] <= 0.0f) && (boundaryParameter[1] < nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5f * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]);

            } else if ((boundaryParameter[0] > 0.0f) && (boundaryParameter[1] >= nodeParameter[1])) {
                integral = (nodeParameter[1] - 0.5f * nodeParameter[1] * nodeParameter[1] / nodeParameter[1]) -
                            (boundaryParameter[0] - 0.5f * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);

            } else if ((boundaryParameter[0] > 0.0f) && (boundaryParameter[1] < nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5f * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]) -
                            (boundaryParameter[0] - 0.5f * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);
            }
        }
    }

    // cleanup
    delete [] nodeParameter;

    return integral;
}
