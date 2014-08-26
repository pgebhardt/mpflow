# --------------------------------------------------------------------
# This file is part of mpFlow.
#
# mpFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# mpFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mpFlow. If not, see <http:#www.gnu.org/licenses/>.
#
# Copyright (C) 2014 Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de
# --------------------------------------------------------------------

from sympy import *
from mathcodegen import *
from kernel import kernel

def integrateOnTriangle(expression, x, y, points):
    # create coordinats
    l1, l2, l3 = symbols('l1, l2, l3')
    l3 = 1 - l1 -l2

    # substitute coordinats
    expression = expression.subs(x,
        l1 * points[0][0] + l2 * points[1][0] + (1 - l1 - l2) * points[2][0])
    expression = expression.subs(y,
        l1 * points[0][1] + l2 * points[1][1] + (1 - l1 - l2) * points[2][1])

    # calc area
    area = 0.5 * Abs(
        (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) -
        (points[2][0] - points[0][0]) * (points[1][1] - points[0][1]))

    # calc integral
    return 2.0 * area * integrate(
        integrate(expression, (l1, 0.0, 1.0 - l2)),
        (l2, 0.0, 1.0))

@symbolic
def coefficients(points, function):
    # get matrix coefficients
    M = []
    for i in range(len(points)):
        N = []
        for j in range(len(points)):
            c = [0.0] * len(points)
            c[j] = 1.0
            N.append(function(points[i], c))
        M.append(N)
    M = Matrix(M)

    # create vector
    V = 1.0 * eye(len(points))

    # calc coefficients
    C = M.LUsolve(V)

    return C.transpose().tolist()

class EdgeBasis(object):
    def __init__(self, name, nodes_per_element, nodes_per_edge,
        basis_function, boundary_function):
        # call base class init
        super(EdgeBasis, self).__init__()

        # save arguments
        self.name = name
        self.nodes_per_element = nodes_per_element
        self.nodes_per_edge = nodes_per_edge
        self.basis_function = basis_function
        self.boundary_function = boundary_function

    @kernel
    def evaluate(self, point, ci, cj):
        # create coordinats
        x, y = symbols('x, y')

        # basis function
        ui = self.basis_function([x, y], ci)
        uj = self.basis_function([x, y], cj)

        return [ui * uj.diff(x) - uj * ui.diff(x),
            ui * uj.diff(y) - uj * ui.diff(y)]

    def render(self, template):
        # arguments
        points_args = [[
            'std::get<0>(this->nodes[{}])'.format(i),
            'std::get<1>(this->nodes[{}])'.format(i)]
            for i in range(self.nodes_per_element)]

        this_coefficients = ['this->coefficients[{}]'.format(i) for i in range(self.nodes_per_element)]
        other_coefficients = ['other->coefficients[{}]'.format(i) for i in range(self.nodes_per_element)]

        # render template
        return template.render(
            # class name
            name=self.name,

            # coefficients in constructor
            coefficients=[
                symbolic(self.basis_function)(
                    ['std::get<0>(this->nodes[node])', 'std::get<1>(this->nodes[node])'],
                    [0.0] * i + [1.0] + [0.0] * (self.nodes_per_element - i - 1))
                for i in range(self.nodes_per_element)],

            # evaluate basis function
            evaluate=self.evaluate(
                ['std::get<{}>(point)'.format(i) for i in range(self.nodes_per_edge)],
                this_coefficients,
                dtype='mpFlow::dtype::real',
                custom_args=['std::tuple<dtype::real, dtype::real> point'],
                name='mpFlow::FEM::basis::{}::evaluate'.format(self.name),
                ),
            )
