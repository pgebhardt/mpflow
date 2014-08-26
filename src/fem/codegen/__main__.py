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

from mako.template import Template
from mathcodegen import symbolic
from basis import Basis, EdgeBasis
import sys, os

def main():
    # load template
    template = Template(filename=os.path.join('codegen', 'templates', 'basis.cpp.mako'))
    edgeTemplate = Template(filename=os.path.join('codegen', 'templates', 'edge_basis.cpp.mako'))

    # create linear basis
    nodeBasises = [
        (Basis(
            name='Linear',
            nodes_per_element=3,
            nodes_per_edge=2,
            basis_function=lambda p, c: c[0] + p[0] * c[1] + p[1] * c[2],
            boundary_function=lambda p, c: c[0] + p * c[1],
            ), template),
        (Basis(
            name='Quadratic',
            nodes_per_element=6,
            nodes_per_edge=3,
            basis_function=lambda p, c: c[0] + p[0] * c[1] + \
                p[1] * c[2] + c[3] * p[0] ** 2 + c[4] * p[1] ** 2 + \
                c[5] * p[0] * p[1],
            boundary_function=lambda p, c: c[0] + p * c[1] + c[2] * p ** 2,
            ), template),
        ]
    basises = []
    basises.append((EdgeBasis(nodeBasises[0][0]), edgeTemplate))

    # render basis functions
    for basis, template in basises:
        print '#' * 50
        print 'render: {}'.format(basis.name)
        file = open('basis_{}.cpp'.format(basis.name.lower()), 'w')
        file.write(basis.render(template))
        file.close()

if __name__ == '__main__':
    main()
