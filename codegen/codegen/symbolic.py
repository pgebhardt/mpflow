from sympy import lambdify, Symbol
from expression import Expression

def symbolic(function):
    def func(*args):
        symargs, expargs = [], []
        args = [arg for arg in args]
        for i in range(len(args)):
            # create expression
            if type(args[i]) in (str, unicode, Expression):
                expargs.append(Expression(args[i]))

            # create symbol
            if type(args[i]) not in (float, int, long):
                symbol = Symbol('tmpsymbol_{}_{}'.format(
                    function.func_name, i))

                symargs.append(symbol)
                args[i] = symbol

        # create lambda function
        lambda_function = lambdify(symargs, function(*args),
            modules=Expression)

        # evaluate expression
        expression = lambda_function(*expargs)

        # check type
        if type(expression) is not Expression:
            expression = Expression(expression)

        return expression

    # save function
    func.function = function

    return func
