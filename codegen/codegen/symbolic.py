from sympy import lambdify, Symbol
from expression import CppExpression

def symbolic(function):
    def func(*args):
        symargs, expargs = [], []
        args = [arg for arg in args]
        for i in range(len(args)):
            # create symbol
            symbol = args[i]
            if not isinstance(args[i], Symbol):
                symbol = Symbol('tmpsymbol_{}'.format(i))

            # create expressions
            expression = args[i]
            if isinstance(args[i], str):
                expression = CppExpression(args[i])
                args[i] = symbol

            # add to lists
            expargs.append(expression)
            symargs.append(symbol)

        # create lambda function
        lambda_function = lambdify(symargs, function(*args),
            modules=CppExpression)

        # evaluate expression
        return lambda_function(*expargs)

    return func
