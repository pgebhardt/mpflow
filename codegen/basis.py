from sympy import *
from mathcodegen import *
from kernel import kernel

def integrateOnTriangle(expression, x, y, points):
    # create coordinats
    l1, l2, l3 = symbols('l1, l2, l3')
    l3 = 1 - l1 -l2

    # substitute coordinats
    expression = expression.subs(x,
        l1 * points[0][0] + l2 * points[1][0] + (1 - l1 - l2) * points[2][0])
    expression = expression.subs(y,
        l1 * points[0][1] + l2 * points[1][1] + (1 - l1 - l2) * points[2][1])

    # calc area
    area = 0.5 * Abs(
        (points[1][0] - points[0][0]) * (points[2][1] - points[0][1]) -
        (points[2][0] - points[0][0]) * (points[1][1] - points[0][1]))

    # calc integral
    return 2.0 * area * integrate(
        integrate(expression, (l1, 0.0, 1.0 - l2)),
        (l2, 0.0, 1.0))

@symbolic
def coefficients(points, function):
    # get matrix coefficients
    M = []
    for i in range(len(points)):
        N = []
        for j in range(len(points)):
            c = [0.0] * len(points)
            c[j] = 1.0
            N.append(function(points[i], c))
        M.append(N + [0.0])
    M = Matrix(M)

    # calc coefficients
    C = []
    c = [Symbol('c_{}'.format(i)) for i in range(len(points))]
    for i in range(len(points)):
        M[i, len(points)] = 1.0
        C.append(solve_linear_system(M, *c).values())
        M[i, len(points)] = 0.0

    return C

class Basis(object):
    def __init__(self, name, nodes_per_element, nodes_per_edge,
        basis_function, boundary_function):
        # call base class init
        super(Basis, self).__init__()

        # save arguments
        self.name = name
        self.nodes_per_element = nodes_per_element
        self.nodes_per_edge = nodes_per_edge
        self.basis_function = basis_function
        self.boundary_function = boundary_function

    @kernel
    def evaluate(self, point, coefficient):
        return self.basis_function(point, coefficient)

    @kernel
    def integrateWithBasis(self, points, ci, cj):
        # create coordinats
        x, y = symbols('x, y')

        # basis function
        ui = self.basis_function([x, y], ci)
        uj = self.basis_function([x, y], cj)

        # integral
        integral = ui * uj

        # integrate on triangle
        return integrateOnTriangle(integral, x, y, points)

    @kernel
    def integrateGradientWithBasis(self, points, ci, cj):
        # create coordinats
        x, y = symbols('x, y')

        # basis function
        ui = self.basis_function([x, y], ci)
        uj = self.basis_function([x, y], cj)

        # integral
        integral = ui.diff(x) * uj.diff(x) + ui.diff(y) * uj.diff(y)

        # integrate on triangle
        return integrateOnTriangle(integral, x, y, points)

    @expressionize
    def integrateBoundaryEdge(self, nodes, coefficients, start, end):
        # integrate boundary_function symbolic
        @symbolic
        def integral(coefficients, start, end):
            x = Symbol('x', real=True)
            return integrate(
                self.boundary_function(x, coefficients),
                (x, start, end))

        # clip integration interval to function definition
        start = start.clip(nodes[0], nodes[self.nodes_per_edge - 1])
        end = end.clip(nodes[0], nodes[self.nodes_per_edge - 1])

        return integral(coefficients, start, end)

    def render(self, template):
        # arguments
        points_args = [[
            'std::get<0>(this->nodes()[{}])'.format(i),
            'std::get<1>(this->nodes()[{}])'.format(i)]
            for i in range(self.nodes_per_element)]

        this_coefficients = ['this->coefficients()[{}]'.format(i) for i in range(self.nodes_per_element)]
        other_coefficients = ['other->coefficients()[{}]'.format(i) for i in range(self.nodes_per_element)]

        # render template
        return template.render(
            # class name
            name=self.name,

            # coefficients in constructor
            coefficients=[
                expressionize(self.basis_function)(
                    ['std::get<0>(this->nodes()[node])', 'std::get<1>(this->nodes()[node])'],
                    [0.0] * i + [1.0] + [0.0] * (self.nodes_per_element - i - 1))
                for i in range(self.nodes_per_element)],

            # evaluate basis function
            evaluate=self.evaluate(
                ['std::get<{}>(point)'.format(i) for i in range(self.nodes_per_edge)],
                this_coefficients,
                dtype='fastEIT::dtype::real',
                custom_args=['std::tuple<dtype::real, dtype::real> point'],
                name='fastEIT::basis::{}::evaluate'.format(self.name),
                ),

            # model integrals
            integrateWithBasis=self.integrateWithBasis(
                points_args, this_coefficients, other_coefficients,
                dtype='fastEIT::dtype::real',
                custom_args=['const std::shared_ptr<{}> other'.format(self.name)],
                name='fastEIT::basis::{}::integrateWithBasis'.format(self.name),
                ),
            integrateGradientWithBasis=self.integrateGradientWithBasis(
                points_args, this_coefficients, other_coefficients,
                dtype='fastEIT::dtype::real',
                custom_args=['const std::shared_ptr<{}> other'.format(self.name)],
                name='fastEIT::basis::{}::integrateGradientWithBasis'.format(self.name),
                ),

            # integrate boundary
            boundaryCoefficiens=coefficients(
                ['nodes[{}]'.format(i) for i in range(self.nodes_per_edge)],
                self.boundary_function),
            integrateBoundaryEdge=self.integrateBoundaryEdge(
                ['nodes[{}]'.format(i) for i in range(self.nodes_per_edge)],
                ['coefficients[{}]'.format(i) for i in range(self.nodes_per_edge)],
                'start', 'end').expand(),
            )
