// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cmath>
#include "fasteit/fasteit.h"

using namespace std;

// create basis class
fastEIT::basis::Linear::Linear(
    std::array<std::tuple<dtype::real, dtype::real>, nodes_per_element> nodes,
    dtype::index one)
    : fastEIT::basis::Basis<nodes_per_edge, nodes_per_element>(nodes, one) {
    // check one
    if (one > nodes_per_element) {
        throw std::invalid_argument(
            "fastEIT::basis::Linear::Linear: one > nodes_per_element");
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
        A[node][0] = (1.0);
        A[node][1] = ((1.0)*(std::get<0>(this->nodes()[node])));
        A[node][2] = ((1.0)*(std::get<1>(this->nodes()[node])));
    }

    // calc coefficients
    this->coefficients() = math::gaussElemination<dtype::real, nodes_per_element>(A, b);
}

// evaluate basis function
fastEIT::dtype::real fastEIT::basis::Linear::evaluate(
    std::tuple<dtype::real, dtype::real> point
    ) {
    return ({((((std::get<0>(point))*(this->coefficients()[1]))+((std::get<1>(point))*(this->coefficients()[2])))+(this->coefficients()[0]));});
}


// integrate with basis
fastEIT::dtype::real fastEIT::basis::Linear::integrateWithBasis(
    const std::shared_ptr<Linear> other
    ) {
    return ({fastEIT::dtype::real subexpression_171744288 = (((((((((((((((((((((((0.0833333333333)*((std::get<0>(this->nodes()[0]))*(std::get<0>(this->nodes()[0]))))*(this->coefficients()[1]))*(other->coefficients()[1]))+(((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*(std::get<1>(this->nodes()[0])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*(std::get<1>(this->nodes()[0])))*(this->coefficients()[2]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*(std::get<0>(this->nodes()[1])))*(this->coefficients()[1]))*(other->coefficients()[1])))+(((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*(std::get<1>(this->nodes()[1])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*(std::get<1>(this->nodes()[1])))*(this->coefficients()[2]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*(std::get<0>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[1])))+(((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[1])))+((((0.166666666667)*(std::get<0>(this->nodes()[0])))*(this->coefficients()[0]))*(other->coefficients()[1])))+((((0.166666666667)*(std::get<0>(this->nodes()[0])))*(this->coefficients()[1]))*(other->coefficients()[0])))+((((0.0833333333333)*((std::get<1>(this->nodes()[0]))*(std::get<1>(this->nodes()[0]))))*(this->coefficients()[2]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*(std::get<0>(this->nodes()[1])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*(std::get<0>(this->nodes()[1])))*(this->coefficients()[2]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<1>(this->nodes()[0])))*(std::get<1>(this->nodes()[1])))*(this->coefficients()[2]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*(std::get<0>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*(std::get<0>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<1>(this->nodes()[0])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[2])))+((((0.166666666667)*(std::get<1>(this->nodes()[0])))*(this->coefficients()[0]))*(other->coefficients()[2])))+((((0.166666666667)*(std::get<1>(this->nodes()[0])))*(this->coefficients()[2]))*(other->coefficients()[0])))+((((0.0833333333333)*((std::get<0>(this->nodes()[1]))*(std::get<0>(this->nodes()[1]))))*(this->coefficients()[1]))*(other->coefficients()[1]));
fastEIT::dtype::real subexpression_172312272 = ((((((((((((((((((((((subexpression_171744288)+(((((0.0833333333333)*(std::get<0>(this->nodes()[1])))*(std::get<1>(this->nodes()[1])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[1])))*(std::get<1>(this->nodes()[1])))*(this->coefficients()[2]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[1])))*(std::get<0>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[1])))+(((((0.0416666666667)*(std::get<0>(this->nodes()[1])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<0>(this->nodes()[1])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[1])))+((((0.166666666667)*(std::get<0>(this->nodes()[1])))*(this->coefficients()[0]))*(other->coefficients()[1])))+((((0.166666666667)*(std::get<0>(this->nodes()[1])))*(this->coefficients()[1]))*(other->coefficients()[0])))+((((0.0833333333333)*((std::get<1>(this->nodes()[1]))*(std::get<1>(this->nodes()[1]))))*(this->coefficients()[2]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<1>(this->nodes()[1])))*(std::get<0>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0416666666667)*(std::get<1>(this->nodes()[1])))*(std::get<0>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<1>(this->nodes()[1])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[2])))+((((0.166666666667)*(std::get<1>(this->nodes()[1])))*(this->coefficients()[0]))*(other->coefficients()[2])))+((((0.166666666667)*(std::get<1>(this->nodes()[1])))*(this->coefficients()[2]))*(other->coefficients()[0])))+((((0.0833333333333)*((std::get<0>(this->nodes()[2]))*(std::get<0>(this->nodes()[2]))))*(this->coefficients()[1]))*(other->coefficients()[1])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[2])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[2])))+(((((0.0833333333333)*(std::get<0>(this->nodes()[2])))*(std::get<1>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[1])))+((((0.166666666667)*(std::get<0>(this->nodes()[2])))*(this->coefficients()[0]))*(other->coefficients()[1])))+((((0.166666666667)*(std::get<0>(this->nodes()[2])))*(this->coefficients()[1]))*(other->coefficients()[0])))+((((0.0833333333333)*((std::get<1>(this->nodes()[2]))*(std::get<1>(this->nodes()[2]))))*(this->coefficients()[2]))*(other->coefficients()[2])))+((((0.166666666667)*(std::get<1>(this->nodes()[2])))*(this->coefficients()[0]))*(other->coefficients()[2])))+((((0.166666666667)*(std::get<1>(this->nodes()[2])))*(this->coefficients()[2]))*(other->coefficients()[0])))+(((0.5)*(this->coefficients()[0]))*(other->coefficients()[0]));
(((1.0)*(subexpression_172312272))*(abs(((((-(std::get<0>(this->nodes()[0])))+(std::get<0>(this->nodes()[1])))*((-(std::get<1>(this->nodes()[0])))+(std::get<1>(this->nodes()[2]))))-(((-(std::get<0>(this->nodes()[0])))+(std::get<0>(this->nodes()[2])))*((-(std::get<1>(this->nodes()[0])))+(std::get<1>(this->nodes()[1]))))))));});
}


// integrate gradient with basis
fastEIT::dtype::real fastEIT::basis::Linear::integrateGradientWithBasis(
    const std::shared_ptr<Linear> other
    ) {
    return ({(((1.0)*((((0.5)*(this->coefficients()[1]))*(other->coefficients()[1]))+(((0.5)*(this->coefficients()[2]))*(other->coefficients()[2]))))*(abs(((((-(std::get<0>(this->nodes()[0])))+(std::get<0>(this->nodes()[1])))*((-(std::get<1>(this->nodes()[0])))+(std::get<1>(this->nodes()[2]))))-(((-(std::get<0>(this->nodes()[0])))+(std::get<0>(this->nodes()[2])))*((-(std::get<1>(this->nodes()[0])))+(std::get<1>(this->nodes()[1]))))))));});
}


// integrate edge
fastEIT::dtype::real fastEIT::basis::Linear::integrateBoundaryEdge(
    std::array<dtype::real, nodes_per_edge> nodes, dtype::index one,
    dtype::real start, dtype::real end) {
    // calc coefficients for basis function
    std::array<dtype::real, nodes_per_edge> coefficients;
    if (one == 0) {
        coefficients[0] = ({(((1.0)*(nodes[1]))/(((-1.0)*(nodes[0]))+((1.0)*(nodes[1]))));});
        coefficients[1] = ({((1.0)/(((1.0)*(nodes[0]))-((1.0)*(nodes[1]))));});
    }
    if (one == 1) {
        coefficients[0] = ({(((1.0)*(nodes[0]))/(((1.0)*(nodes[0]))-((1.0)*(nodes[1]))));});
        coefficients[1] = ({((1.0)/(((-1.0)*(nodes[0]))+((1.0)*(nodes[1]))));});
    }
    return ({(((((-(coefficients[0]))*(min(max((start),(float)(nodes[0])),(float)(nodes[1]))))+((coefficients[0])*(min(max((end),(float)(nodes[0])),(float)(nodes[1])))))-(((coefficients[1])*((min(max((start),(float)(nodes[0])),(float)(nodes[1])))*(min(max((start),(float)(nodes[0])),(float)(nodes[1])))))/(2)))+(((coefficients[1])*((min(max((end),(float)(nodes[0])),(float)(nodes[1])))*(min(max((end),(float)(nodes[0])),(float)(nodes[1])))))/(2)));});
}

// integrate edge with other
fastEIT::dtype::real fastEIT::basis::Linear::integrateBoundaryEdgeWithOther(
    std::array<dtype::real, nodes_per_edge> nodes, dtype::index self,
    dtype::index other, dtype::real start, dtype::real end) {
    // calc coefficients for basis function
    std::array<dtype::real, nodes_per_edge> self_coefficients, other_coefficients;
    if (self == 0) {
        self_coefficients[0] = ({(((1.0)*(nodes[1]))/(((-1.0)*(nodes[0]))+((1.0)*(nodes[1]))));});
        self_coefficients[1] = ({((1.0)/(((1.0)*(nodes[0]))-((1.0)*(nodes[1]))));});
    }
    if (other == 0) {
        other_coefficients[0] = ({(((1.0)*(nodes[1]))/(((-1.0)*(nodes[0]))+((1.0)*(nodes[1]))));});
        other_coefficients[1] = ({((1.0)/(((1.0)*(nodes[0]))-((1.0)*(nodes[1]))));});
    }
    if (self == 1) {
        self_coefficients[0] = ({(((1.0)*(nodes[0]))/(((1.0)*(nodes[0]))-((1.0)*(nodes[1]))));});
        self_coefficients[1] = ({((1.0)/(((-1.0)*(nodes[0]))+((1.0)*(nodes[1]))));});
    }
    if (other == 1) {
        other_coefficients[0] = ({(((1.0)*(nodes[0]))/(((1.0)*(nodes[0]))-((1.0)*(nodes[1]))));});
        other_coefficients[1] = ({((1.0)/(((-1.0)*(nodes[0]))+((1.0)*(nodes[1]))));});
    }
    return ({((((((((-(self_coefficients[0]))*(other_coefficients[0]))*(min(max((start),(float)(nodes[0])),(float)(nodes[1]))))+(((self_coefficients[0])*(other_coefficients[0]))*(min(max((end),(float)(nodes[0])),(float)(nodes[1])))))-((((self_coefficients[1])*(other_coefficients[1]))*((min(max((start),(float)(nodes[0])),(float)(nodes[1])))*((min(max((start),(float)(nodes[0])),(float)(nodes[1])))*(min(max((start),(float)(nodes[0])),(float)(nodes[1]))))))/(3)))+((((self_coefficients[1])*(other_coefficients[1]))*((min(max((end),(float)(nodes[0])),(float)(nodes[1])))*((min(max((end),(float)(nodes[0])),(float)(nodes[1])))*(min(max((end),(float)(nodes[0])),(float)(nodes[1]))))))/(3)))-(((min(max((start),(float)(nodes[0])),(float)(nodes[1])))*(min(max((start),(float)(nodes[0])),(float)(nodes[1]))))*((((self_coefficients[0])*(other_coefficients[1]))/(2))+(((self_coefficients[1])*(other_coefficients[0]))/(2)))))+(((min(max((end),(float)(nodes[0])),(float)(nodes[1])))*(min(max((end),(float)(nodes[0])),(float)(nodes[1]))))*((((self_coefficients[0])*(other_coefficients[1]))/(2))+(((self_coefficients[1])*(other_coefficients[0]))/(2)))));});
}
