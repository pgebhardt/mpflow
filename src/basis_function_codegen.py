from sympy import *
from mako.template import Template
import sys

class CppExpression(object):
    def __init__(self, expression, depth=0, subexpression=None):
        # call base class init
        super(CppExpression, self).__init__()

        # save expression
        self.expression = expression

        # recursion depth
        self.depth = depth;

        # create subexpression
        self.subexpression = subexpression

        # shorten expression
        if depth >= 100:
            self.subexpression = ('subexpression_{}'.format(id(self.expression)),
                CppExpression(self.expression, depth - 1, self.subexpression))
            self.expression = self.subexpression[0]
            self.depth = 0

    def __str__(self):
        return '{}'.format(self.expression)

    def __pos__(self):
        return CppExpression('+{}'.format(self), self.depth + 1, self.subexpression)

    def __neg__(self):
        return CppExpression('-{}'.format(self), self.depth + 1, self.subexpression)

    def __abs__(self):
        return CppExpression('std::abs({})'.format(self), self.depth + 1, self.subexpression)

    def __add__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('({} + {})'.format(self, value), self.depth + 1, self.subexpression)

    def __radd__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('({} + {})'.format(value, self), self.depth + 1, self.subexpression)

    def __sub__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('({} - {})'.format(self, value), self.depth + 1, self.subexpression)

    def __rsub__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('({} - {})'.format(value, self), self.depth + 1, self.subexpression)

    def __mul__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('{} * {}'.format(self, value), self.depth + 1, self.subexpression)

    def __rmul__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('{} * {}'.format(value, self), self.depth + 1, self.subexpression)

    def __truediv__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('{} / {}'.format(self, value), self.depth + 1, self.subexpression)

    def __rtruediv__(self, value):
        if isinstance(value, int):
            value = float(value)

        return CppExpression('{} / {}'.format(value, self), self.depth + 1, self.subexpression)

    def __pow__(self, value):
        if not isinstance(value, int):
            raise RuntimeError('invalid power')

        if value == 1:
            return self

        else:
            return CppExpression('{} * {}'.format(self, self ** (value - 1)), self.depth + 1, self.subexpression)

    def sin(self):
        return CppExpression('sin({})'.format(self), self.depth + 1, self.subexpression)

    def Abs(self):
        return abs(self)


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
    integrals = [ui * uj, ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)]

    # substitute barycentric coordinats
    for f in integrals:
        f = f.subs(x, l1 * x1 + l2 * x2 + (1 - l1 - l2) * x3)
        f = f.subs(y, l1 * y1 + l2 * y2 + (1 - l1 - l2) * y3)

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
            integrals[i] += 'fastEIT::dtype::real {} = {};\n\n'.format(code[0], code[1])

if __name__ == '__main__':
    main()
