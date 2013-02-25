from sympy import lambdify, Symbol
from expression import CppExpression

def symbolic(function):
    def func(*args):
        # create symbols
        symargs, expargs = [], []
        for i in range(len(args)):
            # create expressions
            if isinstance(args[i], str):
                expargs.append(CppExpression(args[i]))
            else:
                expargs.append(args[i])

            # create symbols
            if not isinstance(args[i], Symbol):
                symargs.append(Symbol('tmpsymbol_{}'.format(i)))
            else:
                symargs.append(args[i])

        # lambdify
        lambda_function = lambdify(symargs, function(*symargs),
            modules=CppExpression)

        # evaluate expression
        return lambda_function(*expargs)

    return func
