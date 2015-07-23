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
    Eigen::Ref<Eigen::ArrayXXd const> const points, unsigned const one)
    : mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(points) {
    // check one
    if (one >= pointsPerElement) {
        throw std::invalid_argument(
            "mpFlow::FEM::basis::Linear::Linear: one >= pointsPerElement");
    }

    // calc coefficients with gauss
    std::array<std::array<double, pointsPerElement>, pointsPerElement> A;
    std::array<double, pointsPerElement> b;
    for (unsigned node = 0; node < pointsPerElement; ++node) {
        b[node] = 0.0;
    }
    b[one] = 1.0;

    // fill coefficients
    for (unsigned node = 0; node < pointsPerElement; ++node) {
        A[node][0] = (1.0);
        A[node][1] = ((1.0)*(this->points(node, 0)));
        A[node][2] = ((1.0)*(this->points(node, 1)));
    }

    // calc coefficients
    auto coefficients = math::gaussElemination<double, pointsPerElement>(A, b);
    for (int i = 0; i < this->coefficients.rows(); ++i) {
        this->coefficients(i) = coefficients[i];
    }
}

// evaluate basis function at given point

double mpFlow::FEM::basis::Linear::evaluate(
    Eigen::Ref<Eigen::ArrayXd const> const point
    ) const {
    return ({
((((point(0))*(this->coefficients(1)))+((point(1))*(this->coefficients(2))))+(this->coefficients(0)));
})
;
}


// integrate with basis

double mpFlow::FEM::basis::Linear::integralB(
    Linear const& other
    ) const {
    return ({
(((1.0)*((((((((((((((((((((((((((((((((((((((((((((((0.0833333333333)*((this->points(0, 0))*(this->points(0, 0))))*(this->coefficients(1)))*(other.coefficients(1)))+(((((0.0833333333333)*(this->points(0, 0)))*(this->points(0, 1)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0833333333333)*(this->points(0, 0)))*(this->points(0, 1)))*(this->coefficients(2)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(0, 0)))*(this->points(1, 0)))*(this->coefficients(1)))*(other.coefficients(1))))+(((((0.0416666666667)*(this->points(0, 0)))*(this->points(1, 1)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(0, 0)))*(this->points(1, 1)))*(this->coefficients(2)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(0, 0)))*(this->points(2, 0)))*(this->coefficients(1)))*(other.coefficients(1))))+(((((0.0416666666667)*(this->points(0, 0)))*(this->points(2, 1)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(0, 0)))*(this->points(2, 1)))*(this->coefficients(2)))*(other.coefficients(1))))+((((0.166666666667)*(this->points(0, 0)))*(this->coefficients(0)))*(other.coefficients(1))))+((((0.166666666667)*(this->points(0, 0)))*(this->coefficients(1)))*(other.coefficients(0))))+((((0.0833333333333)*((this->points(0, 1))*(this->points(0, 1))))*(this->coefficients(2)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(0, 1)))*(this->points(1, 0)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(0, 1)))*(this->points(1, 0)))*(this->coefficients(2)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(0, 1)))*(this->points(1, 1)))*(this->coefficients(2)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(0, 1)))*(this->points(2, 0)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(0, 1)))*(this->points(2, 0)))*(this->coefficients(2)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(0, 1)))*(this->points(2, 1)))*(this->coefficients(2)))*(other.coefficients(2))))+((((0.166666666667)*(this->points(0, 1)))*(this->coefficients(0)))*(other.coefficients(2))))+((((0.166666666667)*(this->points(0, 1)))*(this->coefficients(2)))*(other.coefficients(0))))+((((0.0833333333333)*((this->points(1, 0))*(this->points(1, 0))))*(this->coefficients(1)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(1, 0)))*(this->points(1, 1)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0833333333333)*(this->points(1, 0)))*(this->points(1, 1)))*(this->coefficients(2)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(1, 0)))*(this->points(2, 0)))*(this->coefficients(1)))*(other.coefficients(1))))+(((((0.0416666666667)*(this->points(1, 0)))*(this->points(2, 1)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(1, 0)))*(this->points(2, 1)))*(this->coefficients(2)))*(other.coefficients(1))))+((((0.166666666667)*(this->points(1, 0)))*(this->coefficients(0)))*(other.coefficients(1))))+((((0.166666666667)*(this->points(1, 0)))*(this->coefficients(1)))*(other.coefficients(0))))+((((0.0833333333333)*((this->points(1, 1))*(this->points(1, 1))))*(this->coefficients(2)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(1, 1)))*(this->points(2, 0)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0416666666667)*(this->points(1, 1)))*(this->points(2, 0)))*(this->coefficients(2)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(1, 1)))*(this->points(2, 1)))*(this->coefficients(2)))*(other.coefficients(2))))+((((0.166666666667)*(this->points(1, 1)))*(this->coefficients(0)))*(other.coefficients(2))))+((((0.166666666667)*(this->points(1, 1)))*(this->coefficients(2)))*(other.coefficients(0))))+((((0.0833333333333)*((this->points(2, 0))*(this->points(2, 0))))*(this->coefficients(1)))*(other.coefficients(1))))+(((((0.0833333333333)*(this->points(2, 0)))*(this->points(2, 1)))*(this->coefficients(1)))*(other.coefficients(2))))+(((((0.0833333333333)*(this->points(2, 0)))*(this->points(2, 1)))*(this->coefficients(2)))*(other.coefficients(1))))+((((0.166666666667)*(this->points(2, 0)))*(this->coefficients(0)))*(other.coefficients(1))))+((((0.166666666667)*(this->points(2, 0)))*(this->coefficients(1)))*(other.coefficients(0))))+((((0.0833333333333)*((this->points(2, 1))*(this->points(2, 1))))*(this->coefficients(2)))*(other.coefficients(2))))+((((0.166666666667)*(this->points(2, 1)))*(this->coefficients(0)))*(other.coefficients(2))))+((((0.166666666667)*(this->points(2, 1)))*(this->coefficients(2)))*(other.coefficients(0))))+(((0.5)*(this->coefficients(0)))*(other.coefficients(0)))))*(abs(((((-(this->points(0, 0)))+(this->points(1, 0)))*((-(this->points(0, 1)))+(this->points(2, 1))))-(((-(this->points(0, 0)))+(this->points(2, 0)))*((-(this->points(0, 1)))+(this->points(1, 1))))))));
})
;
}


// integrate gradient with basis

double mpFlow::FEM::basis::Linear::integralA(
    Linear const& other
    ) const {
    return ({
(((1.0)*((((0.5)*(this->coefficients(1)))*(other.coefficients(1)))+(((0.5)*(this->coefficients(2)))*(other.coefficients(2)))))*(abs(((((-(this->points(0, 0)))+(this->points(1, 0)))*((-(this->points(0, 1)))+(this->points(2, 1))))-(((-(this->points(0, 0)))+(this->points(2, 0)))*((-(this->points(0, 1)))+(this->points(1, 1))))))));
})
;
}


// integrate edge
double mpFlow::FEM::basis::Linear::boundaryIntegral(
    Eigen::Ref<Eigen::ArrayXd const> const points, unsigned const one,
    double const start, double const end) {
    // calc coefficients for basis function
    std::array<double, pointsPerEdge> coefficients;
    if (one == 0) {
        coefficients[0] = ({
((((1.0)*(points(0)))/(((-1.0)*(points(0)))+((1.0)*(points(1)))))+(1.0));
})
;
        coefficients[1] = ({
((-1.0)/(((-1.0)*(points(0)))+((1.0)*(points(1)))));
})
;
    }
    else
    {
        coefficients[0] = ({
((((1.0)*(points(0)))/(((-1.0)*(points(0)))+((1.0)*(points(1)))))+(1.0));
})
;
        coefficients[1] = ({
((-1.0)/(((-1.0)*(points(0)))+((1.0)*(points(1)))));
})
;
    }
    return ({
(((((-(coefficients[0]))*(min(max((start),(points(0))),(points(1)))))+((coefficients[0])*(min(max((end),(points(0))),(points(1))))))-(((coefficients[1])*((min(max((start),(points(0))),(points(1))))*(min(max((start),(points(0))),(points(1))))))/(2)))+(((coefficients[1])*((min(max((end),(points(0))),(points(1))))*(min(max((end),(points(0))),(points(1))))))/(2)));
})
;
}
