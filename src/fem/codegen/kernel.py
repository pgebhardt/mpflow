# --------------------------------------------------------------------
# This file is part of mpFlow.
#
# mpFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# mpFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with mpFlow. If not, see <http:#www.gnu.org/licenses/>.
#
# Copyright (C) 2014 Patrik Gebhardt
# Contact: patrik.gebhardt@rub.de
# --------------------------------------------------------------------

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

        if type(expression) is list:
            kargs.setdefault('listExpression', expression)

        else:
            kargs.setdefault('expression', expression.expand(dtype=kargs['dtype']))

        # render kernel template
        return template.render(
            args=[arg for arg in args if isinstance(arg, str)],
            **kargs
            )

    # save symbolic
    func.symbolic = sym

    return func
