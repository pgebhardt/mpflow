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

class Basis(object):
    def __init__(self, name, nodes_per_element, nodes_per_edge,
        basis_function, boundary_function):
        # call base class init
        super(Basis, self).__init__()

        # save arguments
        self.name = name
        self.nodes_per_element = nodes_per_element
        self.nodes_per_edge = nodes_per_edge
        self.basis_function = basis_function
        self.boundary_function = boundary_function

    @kernel
    def evaluate(self, point, coefficient):
        return self.basis_function(point, coefficient)

    @kernel
    def integrateWithBasis(self, points, ci, cj):
        # create coordinats
        x, y = symbols('x, y')

        # basis function
        ui = self.basis_function([x, y], ci)
        uj = self.basis_function([x, y], cj)

        # integral
        integral = ui * uj

        # integrate on triangle
        return integrateOnTriangle(integral, x, y, points)

    @kernel
    def integrateGradientWithBasis(self, points, ci, cj):
        # create coordinats
        x, y = symbols('x, y')

        # basis function
        ui = self.basis_function([x, y], ci)
        uj = self.basis_function([x, y], cj)

        # integral
        integral = ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)

        # integrate on triangle
        return integrateOnTriangle(integral, x, y, points)

    @expressionize
    def integrateBoundaryEdge(self, nodes, coefficients, start, end):
        # integrate boundary_function symbolic
        @symbolic
        def integral(coefficients, start, end):
            x = Symbol('x', real=True)
            return integrate(
                self.boundary_function(x, coefficients),
                (x, start, end))

        # clip integration interval to function definition
        start = start.clip(nodes[0], nodes[self.nodes_per_edge - 1])
        end = end.clip(nodes[0], nodes[self.nodes_per_edge - 1])

        return integral(coefficients, start, end)

    def render(self, template):
        # arguments
        points_args = [[
            'this->points({}, 0)'.format(i),
            'this->points({}, 1)'.format(i)]
            for i in range(self.nodes_per_element)]

        this_coefficients = ['this->coefficients({})'.format(i) for i in range(self.nodes_per_element)]
        other_coefficients = ['other->coefficients({})'.format(i) for i in range(self.nodes_per_element)]

        # render template
        return template.render(
            # class name
            name=self.name,

            # coefficients in constructor
            coefficients=[
                symbolic(self.basis_function)(
                    ['this->points(node, 0)', 'this->points(node, 1)'],
                    [0.0] * i + [1.0] + [0.0] * (self.nodes_per_element - i - 1))
                for i in range(self.nodes_per_element)],

            # model integrals
            integrateWithBasis=self.integrateWithBasis(
                points_args, this_coefficients, other_coefficients,
                dtype='double',
                custom_args=['const std::shared_ptr<{}> other'.format(self.name)],
                name='mpFlow::FEM::basis::{}::integrateWithBasis'.format(self.name),
                ),
            integrateGradientWithBasis=self.integrateGradientWithBasis(
                points_args, this_coefficients, other_coefficients,
                dtype='double',
                custom_args=['const std::shared_ptr<{}> other'.format(self.name)],
                name='mpFlow::FEM::basis::{}::integrateGradientWithBasis'.format(self.name),
                ),

            # integrate boundary
            boundaryCoefficiens=coefficients(
                ['points({})'.format(i) for i in range(self.nodes_per_edge)],
                self.boundary_function),
            integrateBoundaryEdge=self.integrateBoundaryEdge(
                ['points({})'.format(i) for i in range(self.nodes_per_edge)],
                ['coefficients[{}]'.format(i) for i in range(self.nodes_per_edge)],
                'start', 'end').expand(),
            )

class EdgeBasis(object):
    def __init__(self, nodeBasis):
        # call base class init
        super(EdgeBasis, self).__init__()

        # save arguments
        self.name = 'Edge'
        self.nodeBasis = nodeBasis

    @kernel
    def evaluate(self, point, length, ci, cj):
        x, y = point

        # basis function
        ui = self.nodeBasis.basis_function([x, y], ci)
        uj = self.nodeBasis.basis_function([x, y], cj)

        return [length * (ui * uj.diff(x) - uj * ui.diff(x)),
            length * (ui * uj.diff(y) - uj * ui.diff(y))]

    @kernel
    def integrateWithBasis(self, points, lengthI, lengthJ, ci1, ci2, cj1, cj2):
        # create coordinats
        x, y = symbols('x, y')

        # create edge based basis functions basis function
        Ni = self.evaluate.symbolic.function(self, [x, y], lengthI, ci1, ci2)
        Nj = self.evaluate.symbolic.function(self, [x, y], lengthJ, cj1, cj2)

        # integrate on triangle
        integrant = Ni[0] * Nj[0] + Ni[1] * Nj[1]
        return integrateOnTriangle(integrant, x, y, points)

    @kernel
    def integrateGradientWithBasis(self, points, lengthI, lengthJ, ci1, ci2, cj1, cj2):
        # create coordinats
        x, y = symbols('x, y')

        # create edge based basis functions basis function
        Ni = self.evaluate.symbolic.function(self, [x, y], lengthI, ci1, ci2)
        Nj = self.evaluate.symbolic.function(self, [x, y], lengthJ, cj1, cj2)

        # integrate on triangle
        integrant = (Ni[1].diff(x) - Ni[0].diff(y)) * (Nj[1].diff(x) - Nj[0].diff(y))
        return integrateOnTriangle(integrant, x, y, points)

    def render(self, template):
        # arguments
        points_args = [[
            'this->points({}, 0)'.format(i),
            'this->points({}, 1)'.format(i)]
            for i in range(self.nodeBasis.nodes_per_element)]

        # coefficients
        ci1 = ['this->nodeBasis[0].coefficients({})'.format(i) for i in range(self.nodeBasis.nodes_per_element)]
        ci2 = ['this->nodeBasis[1].coefficients({})'.format(i) for i in range(self.nodeBasis.nodes_per_element)]
        cj1 = ['other->nodeBasis[0].coefficients({})'.format(i) for i in range(self.nodeBasis.nodes_per_element)]
        cj2 = ['other->nodeBasis[1].coefficients({})'.format(i) for i in range(self.nodeBasis.nodes_per_element)]

        # render template
        return template.render(
            # class name
            name=self.name,

            # model integrals
            integrateWithBasis=self.integrateWithBasis(
                points_args, 'this->length', 'other->length', ci1, ci2, cj1, cj2,
                dtype='double',
                custom_args=['const std::shared_ptr<{}> other'.format(self.name)],
                name='mpFlow::FEM::basis::{}::integrateWithBasis'.format(self.name),
                ),
            integrateGradientWithBasis=self.integrateGradientWithBasis(
                points_args, 'this->length', 'other->length', ci1, ci2, cj1, cj2,
                dtype='double',
                custom_args=['const std::shared_ptr<{}> other'.format(self.name)],
                name='mpFlow::FEM::basis::{}::integrateGradientWithBasis'.format(self.name),
                ),
            )
