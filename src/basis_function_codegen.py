from sympy import *
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
    # sympy symbols
    x1, y1, x2, y2, x3, y3 = symbols('x1, y1, x2, y2, x3, y3')
    ai, bi, ci, di, ei, fi = symbols('ai, bi, ci, di, ei, fi')
    aj, bj, cj, dj, ej, fj = symbols('aj, bj, cj, dj, ej, fj')

    # cpp symbols
    x1_cpp = CppExpression('std::get<0>(self->nodes()[0])')
    y1_cpp = CppExpression('std::get<1>(self->nodes()[0])')
    x2_cpp = CppExpression('std::get<0>(self->nodes()[1])')
    y2_cpp = CppExpression('std::get<1>(self->nodes()[1])')
    x3_cpp = CppExpression('std::get<0>(self->nodes()[2])')
    y3_cpp = CppExpression('std::get<1>(self->nodes()[2])')
    ai_cpp = CppExpression('self->coefficients()[0]')
    bi_cpp = CppExpression('self->coefficients()[1]')
    ci_cpp = CppExpression('self->coefficients()[2]')
    di_cpp = CppExpression('self->coefficients()[3]')
    ei_cpp = CppExpression('self->coefficients()[4]')
    fi_cpp = CppExpression('self->coefficients()[5]')
    aj_cpp = CppExpression('other->coefficients()[0]')
    bj_cpp = CppExpression('other->coefficients()[1]')
    cj_cpp = CppExpression('other->coefficients()[2]')
    dj_cpp = CppExpression('other->coefficients()[3]')
    ej_cpp = CppExpression('other->coefficients()[4]')
    fj_cpp = CppExpression('other->coefficients()[5]')

    # sympy expression
    # area
    sys.setrecursionlimit(10000)
    l1, l2, l3 = symbols('l1, l2, l3')
    x, y = symbols('x, y')
    l3 = 1 - l1 -l2
    
    #ui = ai + bi * x + ci * y + di * x ** 2 + ei * x * y + fi * y ** 2
    #uj = aj + bj * x + cj * y + dj * x ** 2 + ej * x * y + fj * y ** 2
    ui = ai + bi * x + ci * y
    uj = aj + bj * x + cj * y

    f = ui * uj
    fdiff = ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)

    f = f.subs(x, l1 * x1 + l2 * x2 + (1 - l1 - l2) * x3)
    f = f.subs(y, l1 * y1 + l2 * y2 + (1 - l1 - l2) * y3)
    fdiff = fdiff.subs(x, l1 * x1 + l2 * x2 + (1 - l1 - l2) * x3)
    fdiff = fdiff.subs(y, l1 * y1 + l2 * y2 + (1 - l1 - l2) * y3)


    A = 0.5 * Abs((x2 - x1) * (y3 -y1) - (x3 -x1) * (y2 - y1))

    #expression = integrate(f, (l1, 0, 1-l2))
    expression = integrate(fdiff, (l1, 0, 1-l2))
    print 'erster schritt'
    expression= 2 * A * integrate(expression, (l2, 0, 1))
    print 'zweiter schritt'

    # integral
    # expression = a + b * x1 + c * y1
    # create lambda function of expression
    lambda_function = lambdify((x1, y1, x2, y2, x3, y3, ai, bi, ci, di, ei, fi, aj, bj, cj, dj, ej, fj), expression, modules=CppExpression)

    # print evaluated expression
    file = open('integrategradientwithbasis.txt', 'w')
    expression = lambda_function(x1_cpp, y1_cpp, x2_cpp, y2_cpp, x3_cpp, y3_cpp, ai_cpp, bi_cpp, ci_cpp, di_cpp, ei_cpp, fi_cpp, aj_cpp, bj_cpp, cj_cpp, dj_cpp, ej_cpp, fj_cpp)

    cppcode = [('integral', str(expression))]

    while True:
        if expression.subexpression is not None:
            cppcode.append((expression.subexpression[0], str(expression.subexpression[1])))

            expression = expression.subexpression[1]

        else:
            break

    for code in reversed(cppcode):
        file.write('dtype::real {} = {};\n\n'.format(code[0], code[1]))

    file.close()


if __name__ == '__main__':
    main()
