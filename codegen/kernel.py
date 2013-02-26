from sympy import lambdify, Symbol
from mathcodegen import symbolic
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

        # render kernel template
        return template.render(
            args=[arg for arg in args if isinstance(arg, str)],
            expression=expression.expand(dtype=kargs['dtype']),
            **kargs
            )

    # save symbolic
    func.symbolic = sym

    return func
