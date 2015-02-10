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
mpFlow::FEM::basis::Linear::Linear(
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> nodes,
    dtype::index one)
    : mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(nodes) {
    // check one
    if (one >= pointsPerElement) {
        throw std::invalid_argument(
            "mpFlow::FEM::basis::Linear::Linear: one >= pointsPerElement");
    }

    // calc coefficients with gauss
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> A = Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>
        ::Zero(pointsPerElement, pointsPerElement);
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> b = Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1>::Zero(pointsPerElement);
    b(one) = 1.0;

    // fill coefficients
    for (dtype::index node = 0; node < pointsPerElement; ++node) {
        A(node, 0) = (1.0);
        A(node, 1) = ((1.0)*(this->nodes(node, 0)));
        A(node, 2) = ((1.0)*(this->nodes(node, 1)));
    }

    // calc coefficients
    this->coefficients = math::gaussElemination(A, b);
}

// evaluate basis function

mpFlow::dtype::real mpFlow::FEM::basis::Linear::evaluate(
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> point
    ) {
    return ({
((((point(0))*(this->coefficients(1)))+((point(1))*(this->coefficients(2))))+(this->coefficients(0)));
})
;
}


// integrate with basis

mpFlow::dtype::real mpFlow::FEM::basis::Linear::integrateWithBasis(
    const std::shared_ptr<Linear> other
    ) {
    return ({
(((1.0)*((((((((((((((((((((((((((((((((((((((((((((((0.0833333333333)*((this->nodes(0, 0))*(this->nodes(0, 0))))*(this->coefficients(1)))*(other->coefficients(1)))+(((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(0, 1)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(0, 1)))*(this->coefficients(2)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(1, 0)))*(this->coefficients(1)))*(other->coefficients(1))))+(((((0.0416666666667)*(this->nodes(0, 0)))*(this->nodes(1, 1)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(0, 0)))*(this->nodes(1, 1)))*(this->coefficients(2)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(2, 0)))*(this->coefficients(1)))*(other->coefficients(1))))+(((((0.0416666666667)*(this->nodes(0, 0)))*(this->nodes(2, 1)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(0, 0)))*(this->nodes(2, 1)))*(this->coefficients(2)))*(other->coefficients(1))))+((((0.166666666667)*(this->nodes(0, 0)))*(this->coefficients(0)))*(other->coefficients(1))))+((((0.166666666667)*(this->nodes(0, 0)))*(this->coefficients(1)))*(other->coefficients(0))))+((((0.0833333333333)*((this->nodes(0, 1))*(this->nodes(0, 1))))*(this->coefficients(2)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(0, 1)))*(this->nodes(1, 0)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(0, 1)))*(this->nodes(1, 0)))*(this->coefficients(2)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(1, 1)))*(this->coefficients(2)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(0, 1)))*(this->nodes(2, 0)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(0, 1)))*(this->nodes(2, 0)))*(this->coefficients(2)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(2, 1)))*(this->coefficients(2)))*(other->coefficients(2))))+((((0.166666666667)*(this->nodes(0, 1)))*(this->coefficients(0)))*(other->coefficients(2))))+((((0.166666666667)*(this->nodes(0, 1)))*(this->coefficients(2)))*(other->coefficients(0))))+((((0.0833333333333)*((this->nodes(1, 0))*(this->nodes(1, 0))))*(this->coefficients(1)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(1, 1)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(1, 1)))*(this->coefficients(2)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(2, 0)))*(this->coefficients(1)))*(other->coefficients(1))))+(((((0.0416666666667)*(this->nodes(1, 0)))*(this->nodes(2, 1)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(1, 0)))*(this->nodes(2, 1)))*(this->coefficients(2)))*(other->coefficients(1))))+((((0.166666666667)*(this->nodes(1, 0)))*(this->coefficients(0)))*(other->coefficients(1))))+((((0.166666666667)*(this->nodes(1, 0)))*(this->coefficients(1)))*(other->coefficients(0))))+((((0.0833333333333)*((this->nodes(1, 1))*(this->nodes(1, 1))))*(this->coefficients(2)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(1, 1)))*(this->nodes(2, 0)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0416666666667)*(this->nodes(1, 1)))*(this->nodes(2, 0)))*(this->coefficients(2)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(1, 1)))*(this->nodes(2, 1)))*(this->coefficients(2)))*(other->coefficients(2))))+((((0.166666666667)*(this->nodes(1, 1)))*(this->coefficients(0)))*(other->coefficients(2))))+((((0.166666666667)*(this->nodes(1, 1)))*(this->coefficients(2)))*(other->coefficients(0))))+((((0.0833333333333)*((this->nodes(2, 0))*(this->nodes(2, 0))))*(this->coefficients(1)))*(other->coefficients(1))))+(((((0.0833333333333)*(this->nodes(2, 0)))*(this->nodes(2, 1)))*(this->coefficients(1)))*(other->coefficients(2))))+(((((0.0833333333333)*(this->nodes(2, 0)))*(this->nodes(2, 1)))*(this->coefficients(2)))*(other->coefficients(1))))+((((0.166666666667)*(this->nodes(2, 0)))*(this->coefficients(0)))*(other->coefficients(1))))+((((0.166666666667)*(this->nodes(2, 0)))*(this->coefficients(1)))*(other->coefficients(0))))+((((0.0833333333333)*((this->nodes(2, 1))*(this->nodes(2, 1))))*(this->coefficients(2)))*(other->coefficients(2))))+((((0.166666666667)*(this->nodes(2, 1)))*(this->coefficients(0)))*(other->coefficients(2))))+((((0.166666666667)*(this->nodes(2, 1)))*(this->coefficients(2)))*(other->coefficients(0))))+(((0.5)*(this->coefficients(0)))*(other->coefficients(0)))))*(abs(((((-(this->nodes(0, 0)))+(this->nodes(1, 0)))*((-(this->nodes(0, 1)))+(this->nodes(2, 1))))-(((-(this->nodes(0, 0)))+(this->nodes(2, 0)))*((-(this->nodes(0, 1)))+(this->nodes(1, 1))))))));
})
;
}


// integrate gradient with basis

mpFlow::dtype::real mpFlow::FEM::basis::Linear::integrateGradientWithBasis(
    const std::shared_ptr<Linear> other
    ) {
    return ({
(((1.0)*((((0.5)*(this->coefficients(1)))*(other->coefficients(1)))+(((0.5)*(this->coefficients(2)))*(other->coefficients(2)))))*(abs(((((-(this->nodes(0, 0)))+(this->nodes(1, 0)))*((-(this->nodes(0, 1)))+(this->nodes(2, 1))))-(((-(this->nodes(0, 0)))+(this->nodes(2, 0)))*((-(this->nodes(0, 1)))+(this->nodes(1, 1))))))));
})
;
}


// integrate edge
mpFlow::dtype::real mpFlow::FEM::basis::Linear::integrateBoundaryEdge(
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> nodes, dtype::index one,
    dtype::real start, dtype::real end) {
    // calc coefficients for basis function
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> coefficients = Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1>::Zero(pointsPerEdge);
    if (one == 0) {
        coefficients(0) = ({
((((1.0)*(nodes(0)))/(((-1.0)*(nodes(0)))+((1.0)*(nodes(1)))))+(1.0));
})
;
        coefficients(1) = ({
((-1.0)/(((-1.0)*(nodes(0)))+((1.0)*(nodes(1)))));
})
;
    }
    if (one == 1) {
        coefficients(0) = ({
(((-1.0)*(nodes(0)))/(((-1.0)*(nodes(0)))+((1.0)*(nodes(1)))));
})
;
        coefficients(1) = ({
((1.0)/(((-1.0)*(nodes(0)))+((1.0)*(nodes(1)))));
})
;
    }
    return ({
(((((-(coefficients(0)))*(min(max((start),(nodes(0))),(nodes(1)))))+((coefficients(0))*(min(max((end),(nodes(0))),(nodes(1))))))-(((coefficients(1))*((min(max((start),(nodes(0))),(nodes(1))))*(min(max((start),(nodes(0))),(nodes(1))))))/(2)))+(((coefficients(1))*((min(max((end),(nodes(0))),(nodes(1))))*(min(max((end),(nodes(0))),(nodes(1))))))/(2)));
})
;
}
