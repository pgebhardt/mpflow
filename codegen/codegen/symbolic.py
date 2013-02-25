from sympy import lambdify, Symbol
from expression import CppExpression

def symbolic(function):
    def func(*args):
        # create symbols
        symargs, expargs = [], []
        for i in range(len(args)):
            symargs.append(Symbol('tmpsymbol_{}'.format(i)))
            expargs.append(CppExpression(args[i]))

        # lambdify
        lambda_function = lambdify(symargs, function(*symargs),
            modules=CppExpression)

        # evaluate expression
        return lambda_function(*expargs)

    return func
