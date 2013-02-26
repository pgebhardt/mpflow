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

    // calc coefficients
    if (one == 0) {
        this->coefficients()[0] = ((((((-1.0)*(std::get<0>(this->nodes()[0])))*(((((-((-1.0)+(((1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))-(1.0))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))-(((((1.0)*(std::get<1>(this->nodes()[0])))*(((-1.0)+(((1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1]))))))))))/(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))))+(1.0));
        this->coefficients()[1] = (((((-((-1.0)+(((1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))-(1.0))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))));
        this->coefficients()[2] = (((-1.0)+(((1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1]))))))))/(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))));
    }
    if (one == 1) {
        this->coefficients()[0] = (((((-1.0)*(std::get<0>(this->nodes()[0])))*(((1.0)+((((1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/(((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))*(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1]))))))))))))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))+(((((1.0)*(std::get<1>(this->nodes()[0])))*((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))))/(((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))*(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))))));
        this->coefficients()[1] = (((1.0)+((((1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/(((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))*(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1]))))))))))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))));
        this->coefficients()[2] = (((-1.0)*(((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2]))))))/(((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))*(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))));
    }
    if (one == 2) {
        this->coefficients()[0] = (((((1.0)*(std::get<0>(this->nodes()[0])))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/(((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))*(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))))-((((1.0)*(std::get<1>(this->nodes()[0])))/(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))));
        this->coefficients()[1] = (((-1.0)*(((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1]))))))/(((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))*(((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1])))))))))))));
        this->coefficients()[2] = ((1.0)/((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[2])))))-((((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[2])))))*((((-1.0)*(std::get<1>(this->nodes()[0])))+(((1.0)*(std::get<1>(this->nodes()[1])))))))/((((-1.0)*(std::get<0>(this->nodes()[0])))+(((1.0)*(std::get<0>(this->nodes()[1]))))))))));
    }
}

// evaluate basis function
fastEIT::dtype::real fastEIT::basis::Linear::evaluate(
    std::tuple<dtype::real, dtype::real> point
    ) {
    fastEIT::dtype::real result = (((std::get<0>(point))*((this->coefficients()[1])))+(((std::get<1>(point))*((this->coefficients()[2])))))+((this->coefficients()[0]));

    return result;
}


// integrate with basis
fastEIT::dtype::real fastEIT::basis::Linear::integrateWithBasis(
    const std::shared_ptr<Linear> other
    ) {
    fastEIT::dtype::real result = ((1.0)*((((((((((((((((((((((((((((((((((((((((((((((0.0833333333333)*((std::get<0>(this->nodes()[0]))*(std::get<0>(this->nodes()[0]))))*((this->coefficients()[1])))*((other->coefficients()[1])))+((((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*((std::get<1>(this->nodes()[0]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*((std::get<1>(this->nodes()[0]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*((std::get<0>(this->nodes()[1]))))*((this->coefficients()[1])))*((other->coefficients()[1])))))+((((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*((std::get<1>(this->nodes()[1]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*((std::get<1>(this->nodes()[1]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[0])))*((std::get<0>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[1])))))+((((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<0>(this->nodes()[0])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+(((((0.166666666667)*(std::get<0>(this->nodes()[0])))*((this->coefficients()[0])))*((other->coefficients()[1])))))+(((((0.166666666667)*(std::get<0>(this->nodes()[0])))*((this->coefficients()[1])))*((other->coefficients()[0])))))+(((((0.0833333333333)*((std::get<1>(this->nodes()[0]))*(std::get<1>(this->nodes()[0]))))*((this->coefficients()[2])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*((std::get<0>(this->nodes()[1]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*((std::get<0>(this->nodes()[1]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<1>(this->nodes()[0])))*((std::get<1>(this->nodes()[1]))))*((this->coefficients()[2])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*((std::get<0>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<1>(this->nodes()[0])))*((std::get<0>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<1>(this->nodes()[0])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[2])))))+(((((0.166666666667)*(std::get<1>(this->nodes()[0])))*((this->coefficients()[0])))*((other->coefficients()[2])))))+(((((0.166666666667)*(std::get<1>(this->nodes()[0])))*((this->coefficients()[2])))*((other->coefficients()[0])))))+(((((0.0833333333333)*((std::get<0>(this->nodes()[1]))*(std::get<0>(this->nodes()[1]))))*((this->coefficients()[1])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[1])))*((std::get<1>(this->nodes()[1]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[1])))*((std::get<1>(this->nodes()[1]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[1])))*((std::get<0>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[1])))))+((((((0.0416666666667)*(std::get<0>(this->nodes()[1])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<0>(this->nodes()[1])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+(((((0.166666666667)*(std::get<0>(this->nodes()[1])))*((this->coefficients()[0])))*((other->coefficients()[1])))))+(((((0.166666666667)*(std::get<0>(this->nodes()[1])))*((this->coefficients()[1])))*((other->coefficients()[0])))))+(((((0.0833333333333)*((std::get<1>(this->nodes()[1]))*(std::get<1>(this->nodes()[1]))))*((this->coefficients()[2])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<1>(this->nodes()[1])))*((std::get<0>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0416666666667)*(std::get<1>(this->nodes()[1])))*((std::get<0>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<1>(this->nodes()[1])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[2])))))+(((((0.166666666667)*(std::get<1>(this->nodes()[1])))*((this->coefficients()[0])))*((other->coefficients()[2])))))+(((((0.166666666667)*(std::get<1>(this->nodes()[1])))*((this->coefficients()[2])))*((other->coefficients()[0])))))+(((((0.0833333333333)*((std::get<0>(this->nodes()[2]))*(std::get<0>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[1])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[2])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[1])))*((other->coefficients()[2])))))+((((((0.0833333333333)*(std::get<0>(this->nodes()[2])))*((std::get<1>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[1])))))+(((((0.166666666667)*(std::get<0>(this->nodes()[2])))*((this->coefficients()[0])))*((other->coefficients()[1])))))+(((((0.166666666667)*(std::get<0>(this->nodes()[2])))*((this->coefficients()[1])))*((other->coefficients()[0])))))+(((((0.0833333333333)*((std::get<1>(this->nodes()[2]))*(std::get<1>(this->nodes()[2]))))*((this->coefficients()[2])))*((other->coefficients()[2])))))+(((((0.166666666667)*(std::get<1>(this->nodes()[2])))*((this->coefficients()[0])))*((other->coefficients()[2])))))+(((((0.166666666667)*(std::get<1>(this->nodes()[2])))*((this->coefficients()[2])))*((other->coefficients()[0])))))+((((0.5)*(this->coefficients()[0]))*((other->coefficients()[0]))))))*((abs(((((-(std::get<0>(this->nodes()[0])))+((std::get<0>(this->nodes()[1]))))*(((-(std::get<1>(this->nodes()[0])))+((std::get<1>(this->nodes()[2]))))))-((((-(std::get<0>(this->nodes()[0])))+((std::get<0>(this->nodes()[2]))))*(((-(std::get<1>(this->nodes()[0])))+((std::get<1>(this->nodes()[1])))))))))));

    return result;
}


// integrate gradient with basis
fastEIT::dtype::real fastEIT::basis::Linear::integrateGradientWithBasis(
    const std::shared_ptr<Linear> other
    ) {
    fastEIT::dtype::real result = ((1.0)*((((0.5)*(this->coefficients()[1]))*((other->coefficients()[1])))+((((0.5)*(this->coefficients()[2]))*((other->coefficients()[2]))))))*((abs(((((-(std::get<0>(this->nodes()[0])))+((std::get<0>(this->nodes()[1]))))*(((-(std::get<1>(this->nodes()[0])))+((std::get<1>(this->nodes()[2]))))))-((((-(std::get<0>(this->nodes()[0])))+((std::get<0>(this->nodes()[2]))))*(((-(std::get<1>(this->nodes()[0])))+((std::get<1>(this->nodes()[1])))))))))));

    return result;
}


// integrate edge
fastEIT::dtype::real fastEIT::basis::Linear::integrateBoundaryEdge(
    std::array<dtype::real, nodes_per_edge> nodes, dtype::index one,
    dtype::real start, dtype::real end) {
    // crop integration interval to function definition
    start = std::min(std::max(nodes[0], start), nodes[nodes_per_edge - 1]);
    end = std::min(std::max(nodes[0], end), nodes[nodes_per_edge - 1]);

    // calc coefficients for basis function
    std::array<dtype::real, nodes_per_edge> coefficients;
    if (one == 0) {
        coefficients[0] = ((((1.0)*(nodes[0]))/((((-1.0)*(nodes[0]))+(((1.0)*(nodes[1]))))))+(1.0));
        coefficients[1] = ((-1.0)/(((-1.0)*(nodes[0]))+(((1.0)*(nodes[1])))));
    }
    if (one == 1) {
        coefficients[0] = (((-1.0)*(nodes[0]))/((((-1.0)*(nodes[0]))+(((1.0)*(nodes[1]))))));
        coefficients[1] = ((1.0)/(((-1.0)*(nodes[0]))+(((1.0)*(nodes[1])))));
    }

    fastEIT::dtype::real result = ((((-(coefficients[0]))*((start)))+(((coefficients[0])*((end)))))-((((coefficients[1])*(((start)*(start))))/(2))))+((((coefficients[1])*(((end)*(end))))/(2)));

    return result;

}
