from sympy import lambdify, Symbol
from symbolic import symbolic
from mako.template import Template
import os

def kernel(function):
    # load kernel template
    template = Template(
        filename=os.path.join(os.path.dirname(__file__),
        'kernel.mako'))

    # symbolice function
    sym = symbolic(function)

    # kernel
    def func(*args, **kargs):
        # set default kargs
        kargs.setdefault('dtype', 'float')
        kargs.setdefault('header', True)
        kargs.setdefault('custom_args', None)
        kargs.setdefault('name', function.func_name)

        # get expression
        expression = sym(*args)

        # generate subexpressions
        expressions = [
            ('result_{}'.format(function.func_name),
            str(expression)),
            ]
        while True:
            if hasattr(expression, 'subexpression') and \
                expression.subexpression is not None:
                # get subexpression
                expressions.append(
                    (expression.subexpression[0],
                    str(expression.subexpression[1])))

                # go on step deeper
                expression = expression.subexpression[1]
            else:
                break

        # render kernel template
        return template.render(
            args=[arg for arg in args if isinstance(arg, str)],
            expressions=expressions,
            **kargs
            )

    # save symbolic
    func.symbolic = sym

    return func
