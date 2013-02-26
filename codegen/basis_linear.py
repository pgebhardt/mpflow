from mako.template import Template
from basis import *
import sys, os

def main():
    # init sys
    sys.setrecursionlimit(10000)

    # create linear basis
    linear = Basis(
        name='Linear',
        nodes_per_element=3,
        nodes_per_edge=2,
        basis_function=lambda p, c: c[0] + p[0] * c[1] + p[1] * c[2],
        boundary_function=lambda p, c: c[0] + p * c[1],
        )

    # load template
    file = open(os.path.join('src', 'basis_{}.cpp'.format(linear.name.lower())), 'w')
    template = Template(filename=os.path.join('codegen', 'templates', 'basis.cpp.mako'))

    # render template
    file.write(linear.render(template))
    file.close()

if __name__ == '__main__':
    main()
