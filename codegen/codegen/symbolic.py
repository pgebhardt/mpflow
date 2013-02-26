from sympy import lambdify, Symbol
from expression import Expression

def parseArgument(argument, name):
    # string, expression argument
    symbol, expression = None, None
    if type(argument) in (str, unicode, Expression):
        # create symbols
        expression = Expression(argument)
        symbol = Symbol(name, real=True)

        # replace argument by symbol
        argument = symbol

    elif type(argument) is list:
        newarg, symbol, expression = [], [], []
        for i in range(len(argument)):
            # parse argument
            res = parseArgument(argument[i],
                '{}_{}'.format(name, i))

            # add args to lists
            newarg.append(res[0])
            if res[1] is not None:
                symbol += res[1] if type(res[1]) is list else [res[1]]
            if res[2] is not None:
                expression += res[2] if type(res[2]) is list else [res[2]]

        # replace argument by symbols
        argument = newarg
    return argument, symbol, expression

def symbolic(function):
    def func(*args):
        # parse arguments
        args = [arg for arg in args]
        args, symargs, expargs = parseArgument(args, 'tempsymbol')

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
