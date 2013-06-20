from mako.template import Template
from mathcodegen import symbolic
from basis import *
import sys, os

def main():
    # create linear basis
    basises = [
        Basis(
            name='Linear',
            nodes_per_element=3,
            nodes_per_edge=2,
            basis_function=lambda p, c: c[0] + p[0] * c[1] + p[1] * c[2],
            boundary_function=lambda p, c: c[0] + p * c[1],
            ),
        Basis(
            name='Quadratic',
            nodes_per_element=6,
            nodes_per_edge=3,
            basis_function=lambda p, c: c[0] + p[0] * c[1] + \
                p[1] * c[2] + c[3] * p[0] ** 2 + c[4] * p[1] ** 2 + \
                c[5] * p[0] * p[1],
            boundary_function=lambda p, c: c[0] + p * c[1] + c[2] * p ** 2,
            ),
        ]

    # load template
    template = Template(filename=os.path.join('codegen', 'templates', 'basis.cpp.mako'))

    # render basis functions
    for basis in basises:
        print '#' * 50
        print 'render: {}'.format(basis.name)
        file = open(os.path.join('src', 'eit', 'basis_{}.cpp'.format(basis.name.lower())), 'w')
        file.write(basis.render(template))
        file.close()

if __name__ == '__main__':
    main()
