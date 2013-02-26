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
def basis(x, y, a, b, c):
    return a + b * x + c * y

@kernel
def integrateWithBasis(x1, y1, x2, y2, x3, y3, ai, bi, ci, aj, bj, cj):
    # create coordinats
    x, y = symbols('x, y')

    # basis function
    ui = basis.symbolic.function(x, y, ai, bi, ci)
    uj = basis.symbolic.function(x, y, aj, bj, cj)

    # integral
    integral = ui * uj

    # integrate on triangle
    return integrateOnTriangle(integral, x, y,
        [[x1, y1], [x2, y2], [x3, y3]])

@kernel
def integrateGradientWithBasis(x1, y1, x2, y2, x3, y3, ai, bi, ci, aj, bj, cj):
    # create coordinats
    x, y = symbols('x, y')

    # basis function
    ui = basis.symbolic.function(x, y, ai, bi, ci)
    uj = basis.symbolic.function(x, y, aj, bj, cj)

    # integral
    integral = ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)

    # integrate on triangle
    return integrateOnTriangle(integral, x, y,
        [[x1, y1], [x2, y2], [x3, y3]])

@kernel
def integrateBoundaryEdge(a, b, start, end):
    # create coordinats
    x = Symbol('x')

    return integrate(a + b * x, (x, start, end))

def main():
    # init sys
    sys.setrecursionlimit(10000)

    # arguments
    args = [
        'std::get<0>(this->nodes()[0])',
        'std::get<1>(this->nodes()[0])',
        'std::get<0>(this->nodes()[1])',
        'std::get<1>(this->nodes()[1])',
        'std::get<0>(this->nodes()[2])',
        'std::get<1>(this->nodes()[2])',
        'this->coefficients()[0]',
        'this->coefficients()[1]',
        'this->coefficients()[2]',
        'other->coefficients()[0]',
        'other->coefficients()[1]',
        'other->coefficients()[2]',
        ]

    # apply to template
    file = open(os.path.join('src', 'basis_linear.cpp'), 'w')
    template = Template(filename=os.path.join('codegen', 'templates', 'basis_linear.cpp.mako'))
    file.write(template.render(
        coefficients=[
            basis.symbolic(
                'std::get<0>(this->nodes()[node])',
                'std::get<1>(this->nodes()[node])',
                1.0, 0.0, 0.0),
            basis.symbolic(
                'std::get<0>(this->nodes()[node])',
                'std::get<1>(this->nodes()[node])',
                0.0, 1.0, 0.0),
            basis.symbolic(
                'std::get<0>(this->nodes()[node])',
                'std::get<1>(this->nodes()[node])',
                0.0, 0.0, 1.0),
            ],
        evaluate=basis(*[
                'std::get<0>(point)',
                'std::get<1>(point)',
                'this->coefficients()[0]',
                'this->coefficients()[1]',
                'this->coefficients()[2]',
            ],
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
        integrateBoundaryEdge=integrateBoundaryEdge(
            'coefficients[0]', 'coefficients[1]', 'start', 'end',
            dtype='fastEIT::dtype::real', header=False,
            ),
        ))
    file.close()

if __name__ == '__main__':
    main()
