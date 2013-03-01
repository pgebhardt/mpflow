from mako.template import Template
from mathcodegen import symbolic
from basis import *
import sys, os

def main():
    # init sys
    sys.setrecursionlimit(10000)

    # create quadratic basis
    quadratic = Basis(
        name='Quadratic',
        nodes_per_element=6,
        nodes_per_edge=3,
        basis_function=lambda p, c: c[0] + p[0] * c[1] + p[1] * c[2] + c[3] * p[0] ** 2 + c[4] * p[1] ** 2 + c[5] * p[0] * p[1],
        boundary_function=lambda p, c: c[0] + p * c[1] + c[2] * p ** 2,
        )

    # load template
    file = open(os.path.join('src', 'basis_{}.cpp'.format(quadratic.name.lower())), 'w')
    template = Template(filename=os.path.join('codegen', 'templates', 'basis.cpp.mako'))

    # render template
    file.write(quadratic.render(template))
    file.close()

if __name__ == '__main__':
    main()
