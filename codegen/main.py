from sympy import *
from mako.template import Template
from expression import CppExpression
import sys
import os

def main():
    # init sys
    sys.setrecursionlimit(10000)

    # sympy symbols
    x1, y1, x2, y2, x3, y3 = symbols('x1, y1, x2, y2, x3, y3')
    ai, bi, ci = symbols('ai, bi, ci')
    aj, bj, cj = symbols('aj, bj, cj')
    l1, l2, l3 = symbols('l1, l2, l3')
    x, y = symbols('x, y')
    l3 = 1 - l1 -l2

    # cpp symbols
    x1_cpp = CppExpression('std::get<0>(this->nodes()[0])')
    y1_cpp = CppExpression('std::get<1>(this->nodes()[0])')
    x2_cpp = CppExpression('std::get<0>(this->nodes()[1])')
    y2_cpp = CppExpression('std::get<1>(this->nodes()[1])')
    x3_cpp = CppExpression('std::get<0>(this->nodes()[2])')
    y3_cpp = CppExpression('std::get<1>(this->nodes()[2])')
    ai_cpp = CppExpression('this->coefficients()[0]')
    bi_cpp = CppExpression('this->coefficients()[1]')
    ci_cpp = CppExpression('this->coefficients()[2]')
    aj_cpp = CppExpression('other->coefficients()[0]')
    bj_cpp = CppExpression('other->coefficients()[1]')
    cj_cpp = CppExpression('other->coefficients()[2]')

    # basis function
    ui = ai + bi * x + ci * y
    uj = aj + bj * x + cj * y

    # integrals to calculate
    integrals = [
        ui * uj,
        ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y),
        ]

    # substitute barycentric coordinats
    for i in range(len(integrals)):
        integrals[i] = integrals[i].subs(x, l1 * x1 + l2 * x2 + (1 - l1 - l2) * x3)
        integrals[i] = integrals[i].subs(y, l1 * y1 + l2 * y2 + (1 - l1 - l2) * y3)

    # area
    area = 0.5 * Abs((x2 - x1) * (y3 -y1) - (x3 -x1) * (y2 - y1))

    # integrate
    for i in range(len(integrals)):
        integrals[i] = 2.0 * area * integrate(
            integrate(integrals[i], (l1, 0.0, 1.0 - l2)),
            (l2, 0.0, 1.0))

    # create cpp expressions
    for i in range(len(integrals)):
        integrals[i] = lambdify((x1, y1, x2, y2, x3, y3, ai, bi, ci, aj, bj, cj),
            integrals[i], modules=CppExpression)

    # evaluate expressions
    for i in range(len(integrals)):
        # print evaluated expression
        expression = integrals[i](
            x1_cpp, y1_cpp, x2_cpp,
            y2_cpp, x3_cpp, y3_cpp,
            ai_cpp, bi_cpp, ci_cpp,
            aj_cpp, bj_cpp, cj_cpp,
            )

        # generate cpp code
        cppcode = [('integral', str(expression))]
        while True:
            if expression.subexpression is not None:
                cppcode.append((expression.subexpression[0], str(expression.subexpression[1])))

                expression = expression.subexpression[1]

            else:
                break

        # write string to list
        integrals[i] = ''
        for code in reversed(cppcode):
            integrals[i] += 'fastEIT::dtype::real {} = {};\n'.format(code[0], code[1])

    # apply to template
    file = open(os.path.join('src', 'basis_linear.cpp'), 'w')
    template = Template(filename=os.path.join('codegen', 'templates', 'basis_linear.cpp.mako'))
    file.write(template.render(
        integrateWithBasis=integrals[0],
        integrateGradientWithBasis=integrals[1],
        ))
    file.close()

if __name__ == '__main__':
    main()
