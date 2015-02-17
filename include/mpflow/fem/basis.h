// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLUDE_FEM_BASIS_H
#define MPFLOW_INCLUDE_FEM_BASIS_H

// namespace mpFlow::FEM::basis
namespace mpFlow {
namespace FEM {
namespace basis {
    // abstract basis class
    template <
        int _pointsPerEdge,
        int _pointsPerElement
    >
    class Basis {
    // constructor and destructor
    protected:
        Basis(std::array<std::tuple<dtype::real, dtype::real>, _pointsPerElement> points)
            : nodes(points) {
            for (auto& coefficient : this->coefficients) {
                coefficient = 0.0f;
            }
        }

        virtual ~Basis() { }

    public:
        // geometry definition
        static const dtype::size pointsPerEdge = _pointsPerEdge;
        static const dtype::size pointsPerElement = _pointsPerElement;

        // member
        std::array<std::tuple<dtype::real, dtype::real>, pointsPerElement> nodes;
        std::array<dtype::real, pointsPerElement> coefficients;
    };

    // linear basis class definition
    class Linear : public Basis<2, 3> {
    public:
        // constructor
        Linear(std::array<std::tuple<dtype::real, dtype::real>, pointsPerElement> nodes,
            dtype::index one);

        // mathematical evaluation of basis
        dtype::real integrateWithBasis(const std::shared_ptr<Linear> other);
        dtype::real integrateGradientWithBasis(const std::shared_ptr<Linear> other);
        static dtype::real integrateBoundaryEdge(
            std::array<dtype::real, pointsPerEdge> nodes, dtype::index one,
            dtype::real start, dtype::real end);

        // evaluation
        dtype::real evaluate(std::tuple<dtype::real, dtype::real> point);
    };

    // quadratic basis class definition
    class Quadratic : public Basis<3, 6> {
    public:
        // constructor
        Quadratic(std::array<std::tuple<dtype::real, dtype::real>, pointsPerElement> nodes,
            dtype::index one);

        // mathematical evaluation of basis
        dtype::real integrateWithBasis(const std::shared_ptr<Quadratic> other);
        dtype::real integrateGradientWithBasis(const std::shared_ptr<Quadratic> other);
        static dtype::real integrateBoundaryEdge(
            std::array<dtype::real, pointsPerEdge> nodes, dtype::index one,
            dtype::real start, dtype::real end);

        // evaluation
        dtype::real evaluate(std::tuple<dtype::real, dtype::real> point);
    };

    // edge bases basis function definition
    class Edge : public Basis<1, 3> {
    public:
        // constructor
        Edge(std::array<std::tuple<dtype::real, dtype::real>, Linear::pointsPerElement> nodes,
            std::tuple<dtype::index, dtype::index> edge);

        // mathematical evaluation of basis
        dtype::real integrateWithBasis(const std::shared_ptr<Edge> other);
        dtype::real integrateGradientWithBasis(const std::shared_ptr<Edge> other);
        std::tuple<dtype::real, dtype::real> evaluate(std::tuple<dtype::real, dtype::real> point);

        // member
        std::array<Linear, 2> nodeBasis;
        dtype::real length;
    };
}
}
}

#endif
