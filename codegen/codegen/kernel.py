from sympy import lambdify, Symbol
from expression import CppExpression
from symbolic import symbolic
from mako.template import Template
import os

def kernel(dtype='float', header=True, custom_args=None, name=''):
    # load kernel template
    template = Template(
        filename=os.path.join(os.path.dirname(__file__),
        'kernel.mako'))
    # decorator
    def decorator(function):
        # symbolice function
        sym = symbolic(function)

        # kernel
        def func(*args):
            # get expression
            expression = sym(*args)

            # generate subexpressions
            subexpressions = [
                ('result_{}'.format(function.func_name),
                str(expression)),
                ]
            while True:
                if expression.subexpression is not None:
                    subexpressions.append(
                        (expression.subexpression[0],
                        str(expression.subexpression[1])))

                    expression = expression.subexpression[1]

                else:
                    break

            return template.render(
                dtype=dtype,
                header=header,
                name=name,
                custom_args=custom_args,
                args=args,
                subexpressions=subexpressions,
                )

        return func
    return decorator
