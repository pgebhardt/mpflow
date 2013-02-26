from sympy import *
from mako.template import Template
from codegen import *
import sys
import os

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

@kernel
def basis(point, coefficient):
    return coefficient[0] + coefficient[1] * point[0] + coefficient[2] * point[1]

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
    V = Matrix([0.0] * len(points))

    # calc coefficients
    result = []
    for i in range(len(points)):
        V[i] = 1.0
        C = M.inv() * V
        result.append([c for c in C])
        V[i] = 0.0

    return result

@kernel
def integrateWithBasis(points, ci, cj):
    # create coordinats
    x, y = symbols('x, y')

    # basis function
    ui = basis.symbolic.function([x, y], ci)
    uj = basis.symbolic.function([x, y], cj)

    # integral
    integral = ui * uj

    # integrate on triangle
    return integrateOnTriangle(integral, x, y, points)

@kernel
def integrateGradientWithBasis(points, ci, cj):
    # create coordinats
    x, y = symbols('x, y')

    # basis function
    ui = basis.symbolic.function([x, y], ci)
    uj = basis.symbolic.function([x, y], cj)

    # integral
    integral = ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)

    # integrate on triangle
    return integrateOnTriangle(integral, x, y, points)

@kernel
def integrateBoundaryEdge(a, b, start, end):
    # create coordinats
    x = Symbol('x')

    return integrate(a + b * x, (x, start, end))

def main():
    # init sys
    sys.setrecursionlimit(10000)

    # arguments
    args = [[
        ['std::get<0>(this->nodes()[0])', 'std::get<1>(this->nodes()[0])'],
        ['std::get<0>(this->nodes()[1])', 'std::get<1>(this->nodes()[1])'],
        ['std::get<0>(this->nodes()[2])', 'std::get<1>(this->nodes()[2])']],
        ['this->coefficients()[0]', 'this->coefficients()[1]', 'this->coefficients()[2]'],
        ['other->coefficients()[0]', 'other->coefficients()[1]', 'other->coefficients()[2]'],
        ]

    # apply to template
    file = open(os.path.join('src', 'basis_linear.cpp'), 'w')
    template = Template(filename=os.path.join('codegen', 'templates', 'basis_linear.cpp.mako'))
    file.write(template.render(
        coefficients=coefficients(args[0], basis.symbolic.function),
        evaluate=basis(['std::get<0>(point)', 'std::get<1>(point)'],
            ['this->coefficients()[0]', 'this->coefficients()[1]', 'this->coefficients()[2]'],
            dtype='fastEIT::dtype::real',
            custom_args=['std::tuple<dtype::real, dtype::real> point'],
            name='fastEIT::basis::Linear::evaluate'
            ),
        integrateWithBasis=integrateWithBasis(*args,
            dtype='fastEIT::dtype::real',
            custom_args=['const std::shared_ptr<Linear> other'],
            name='fastEIT::basis::Linear::integrateWithBasis'
            ),
        integrateGradientWithBasis=integrateGradientWithBasis(*args,
            dtype='fastEIT::dtype::real',
            custom_args=['const std::shared_ptr<Linear> other'],
            name='fastEIT::basis::Linear::integrateGradientWithBasis'
            ),
        boundaryCoefficiens=coefficients(
            ['nodes[0]', 'nodes[1]'], lambda x, c: c[0] + x * c[1]),
        integrateBoundaryEdge=integrateBoundaryEdge(
            'coefficients[0]', 'coefficients[1]', 'start', 'end',
            dtype='fastEIT::dtype::real', header=False,
            ),
        ))
    file.close()

if __name__ == '__main__':
    main()
