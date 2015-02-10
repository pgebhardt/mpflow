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
mpFlow::FEM::basis::Edge::Edge (
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic> nodes,
    std::tuple<dtype::index, dtype::index> edge) :
    mpFlow::FEM::basis::Basis<pointsPerEdge, pointsPerElement>(nodes),
    nodeBasis({{ Linear(nodes, std::get<0>(edge)), Linear(nodes, std::get<1>(edge)) }}) {
    // calculate length of edge
    this->length = sqrt(math::square(nodes(std::get<1>(edge), 0) - nodes(std::get<0>(edge), 0)) +
                        math::square(nodes(std::get<1>(edge), 1) - nodes(std::get<0>(edge), 1)));
}

// evaluate basis function

std::tuple<mpFlow::dtype::real, mpFlow::dtype::real> mpFlow::FEM::basis::Edge::evaluate(
    Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, 1> point
    ) {
    return std::make_tuple(
        ({
((this->length)*(((-(this->nodeBasis[0].coefficients(1)))*((((point(0))*(this->nodeBasis[1].coefficients(1)))+((point(1))*(this->nodeBasis[1].coefficients(2))))+(this->nodeBasis[1].coefficients(0))))+((this->nodeBasis[1].coefficients(1))*((((point(0))*(this->nodeBasis[0].coefficients(1)))+((point(1))*(this->nodeBasis[0].coefficients(2))))+(this->nodeBasis[0].coefficients(0))))));
})
,
        ({
((this->length)*(((-(this->nodeBasis[0].coefficients(2)))*((((point(0))*(this->nodeBasis[1].coefficients(1)))+((point(1))*(this->nodeBasis[1].coefficients(2))))+(this->nodeBasis[1].coefficients(0))))+((this->nodeBasis[1].coefficients(2))*((((point(0))*(this->nodeBasis[0].coefficients(1)))+((point(1))*(this->nodeBasis[0].coefficients(2))))+(this->nodeBasis[0].coefficients(0))))));
})

        );
}


// integrate with basis

mpFlow::dtype::real mpFlow::FEM::basis::Edge::integrateWithBasis(
    const std::shared_ptr<Edge> other
    ) {
    return ({
(((1.0)*(((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((0.0833333333333)*((this->nodes(0, 0))*(this->nodes(0, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2)))-((((((((0.0833333333333)*((this->nodes(0, 0))*(this->nodes(0, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.0833333333333)*((this->nodes(0, 0))*(this->nodes(0, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.0833333333333)*((this->nodes(0, 0))*(this->nodes(0, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((((0.0833333333333)*(this->nodes(0, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))-((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(0, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.0833333333333)*((this->nodes(0, 1))*(this->nodes(0, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.0833333333333)*((this->nodes(0, 1))*(this->nodes(0, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.0833333333333)*((this->nodes(0, 1))*(this->nodes(0, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.0833333333333)*((this->nodes(0, 1))*(this->nodes(0, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((((0.0833333333333)*(this->nodes(0, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(0, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.0833333333333)*((this->nodes(1, 0))*(this->nodes(1, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.0833333333333)*((this->nodes(1, 0))*(this->nodes(1, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.0833333333333)*((this->nodes(1, 0))*(this->nodes(1, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.0833333333333)*((this->nodes(1, 0))*(this->nodes(1, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+(((((((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-(((((((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((((0.0833333333333)*(this->nodes(1, 0)))*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))-((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(1, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.0833333333333)*((this->nodes(1, 1))*(this->nodes(1, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.0833333333333)*((this->nodes(1, 1))*(this->nodes(1, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.0833333333333)*((this->nodes(1, 1))*(this->nodes(1, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.0833333333333)*((this->nodes(1, 1))*(this->nodes(1, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+(((((((((0.0833333333333)*(this->nodes(1, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-(((((((((0.0833333333333)*(this->nodes(1, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((((0.0833333333333)*(this->nodes(1, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((((0.0833333333333)*(this->nodes(1, 1)))*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(1, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.0833333333333)*((this->nodes(2, 0))*(this->nodes(2, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.0833333333333)*((this->nodes(2, 0))*(this->nodes(2, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.0833333333333)*((this->nodes(2, 0))*(this->nodes(2, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.0833333333333)*((this->nodes(2, 0))*(this->nodes(2, 0))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))-((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(2, 0)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.0833333333333)*((this->nodes(2, 1))*(this->nodes(2, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.0833333333333)*((this->nodes(2, 1))*(this->nodes(2, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.0833333333333)*((this->nodes(2, 1))*(this->nodes(2, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.0833333333333)*((this->nodes(2, 1))*(this->nodes(2, 1))))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))-((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))+((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))-((((((((0.166666666667)*(this->nodes(2, 1)))*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))-(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))+(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))-(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(0)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0))))-(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(1))))+(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(0))))-(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(0)))*(other->nodeBasis[1].coefficients(2))))+(((((((0.5)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(0)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(0)))))*(abs(((((-(this->nodes(0, 0)))+(this->nodes(1, 0)))*((-(this->nodes(0, 1)))+(this->nodes(2, 1))))-(((-(this->nodes(0, 0)))+(this->nodes(2, 0)))*((-(this->nodes(0, 1)))+(this->nodes(1, 1))))))));
})
;
}


// integrate gradient with basis

mpFlow::dtype::real mpFlow::FEM::basis::Edge::integrateGradientWithBasis(
    const std::shared_ptr<Edge> other
    ) {
    return ({
(((1.0)*((((((((((2.0)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2)))-(((((((2.0)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(1)))*(this->nodeBasis[1].coefficients(2)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1))))-(((((((2.0)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(1)))*(other->nodeBasis[1].coefficients(2))))+(((((((2.0)*(this->length))*(other->length))*(this->nodeBasis[0].coefficients(2)))*(this->nodeBasis[1].coefficients(1)))*(other->nodeBasis[0].coefficients(2)))*(other->nodeBasis[1].coefficients(1)))))*(abs(((((-(this->nodes(0, 0)))+(this->nodes(1, 0)))*((-(this->nodes(0, 1)))+(this->nodes(2, 1))))-(((-(this->nodes(0, 0)))+(this->nodes(2, 0)))*((-(this->nodes(0, 1)))+(this->nodes(1, 1))))))));
})
;
}

