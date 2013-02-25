from sympy import *
from mako.template import Template
from codegen import *
import sys
import os

@kernel(dtype='fastEIT::dtype::real',
    name='fastEIT::basis::Linear::integrateWithBasis',
    custom_args=['const std::shared_ptr<Linear> other'])
def integrateWithBasis(x1, y1, x2, y2, x3, y3, ai, bi, ci, aj, bj, cj):
    # create coordinats
    x, y = symbols('x, y')
    l1, l2, l3 = symbols('l1, l2, l3')
    l3 = 1 - l1 -l2

    # basis function
    ui = ai + bi * x + ci * y
    uj = aj + bj * x + cj * y

    # integral
    integral = ui * uj

    # substitute coordinats
    integral = integral.subs(x, l1 * x1 + l2 * x2 + (1 - l1 - l2) * x3)
    integral = integral.subs(y, l1 * y1 + l2 * y2 + (1 - l1 - l2) * y3)

    # calc area
    area = 0.5 * Abs((x2 - x1) * (y3 -y1) - (x3 -x1) * (y2 - y1))

    # calc integral
    return 2.0 * area * integrate(
        integrate(integral, (l1, 0.0, 1.0 - l2)),
        (l2, 0.0, 1.0))

@kernel(dtype='fastEIT::dtype::real',
    name='fastEIT::basis::Linear::integrateGradientWithBasis',
    custom_args=['const std::shared_ptr<Linear> other'])
def integrateGradientWithBasis(x1, y1, x2, y2, x3, y3, ai, bi, ci, aj, bj, cj):
    # create coordinats
    x, y = symbols('x, y')
    l1, l2, l3 = symbols('l1, l2, l3')
    l3 = 1 - l1 -l2

    # basis function
    ui = ai + bi * x + ci * y
    uj = aj + bj * x + cj * y

    # integral
    integral = ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)

    # substitute coordinats
    integral = integral.subs(x, l1 * x1 + l2 * x2 + (1 - l1 - l2) * x3)
    integral = integral.subs(y, l1 * y1 + l2 * y2 + (1 - l1 - l2) * y3)

    # calc area
    area = 0.5 * Abs((x2 - x1) * (y3 -y1) - (x3 -x1) * (y2 - y1))

    # calc integral
    return 2.0 * area * integrate(
        integrate(integral, (l1, 0.0, 1.0 - l2)),
        (l2, 0.0, 1.0))

@kernel(dtype='fastEIT::dtype::real',
    header=False)
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
        integrateWithBasis=integrateWithBasis(*args),
        integrateGradientWithBasis=integrateGradientWithBasis(*args),
        integrateBoundaryEdge=integrateBoundaryEdge(
            'coefficients[0]',
            'coefficients[1]',
            'start',
            'end',
            ),
        ))
    file.close()

if __name__ == '__main__':
    main()
