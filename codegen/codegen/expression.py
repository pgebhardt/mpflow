class Expression(object):
    def __init__(self, expression, depth=0, subexpression=None):
        # call base class init
        super(Expression, self).__init__()

        # save expression
        self.expression = expression

        # recursion depth
        self.depth = depth;

        # create subexpression
        self.subexpression = subexpression

        # shorten expression
        if depth >= 100:
            self.subexpression = ('subexpression_{}'.format(id(self.expression)),
                Expression(self.expression, depth - 1, self.subexpression))
            self.expression = self.subexpression[0]
            self.depth = 0

    def __str__(self):
        return '({})'.format(self.expression)

    def __pos__(self):
        return Expression('+{}'.format(self), self.depth + 1, self.subexpression)

    def __neg__(self):
        return Expression('-{}'.format(self), self.depth + 1, self.subexpression)

    def __abs__(self):
        return Expression('std::abs({})'.format(self), self.depth + 1, self.subexpression)

    def __add__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} + {}'.format(self, value), self.depth + 1, self.subexpression)

    def __radd__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} + {}'.format(value, self), self.depth + 1, self.subexpression)

    def __sub__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} - {}'.format(self, value), self.depth + 1, self.subexpression)

    def __rsub__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} - {}'.format(value, self), self.depth + 1, self.subexpression)

    def __mul__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} * {}'.format(self, value), self.depth + 1, self.subexpression)

    def __rmul__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} * {}'.format(value, self), self.depth + 1, self.subexpression)

    def __truediv__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} / {}'.format(self, value), self.depth + 1, self.subexpression)

    def __rtruediv__(self, value):
        if isinstance(value, int):
            value = float(value)

        return Expression('{} / {}'.format(value, self), self.depth + 1, self.subexpression)

    def __pow__(self, value):
        if not isinstance(value, int):
            raise RuntimeError('invalid power')

        if value == 1:
            return self

        else:
            return Expression('{} * {}'.format(self, self ** (value - 1)), self.depth + 1, self.subexpression)

    def sin(self):
        return Expression('sin({})'.format(self), self.depth + 1, self.subexpression)

    def Abs(self):
        return abs(self)
