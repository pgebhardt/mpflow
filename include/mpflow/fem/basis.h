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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLUDE_FEM_BASIS_H
#define MPFLOW_INCLUDE_FEM_BASIS_H

#include <functional>

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
        Basis(Eigen::Ref<Eigen::ArrayXXd const> const points)
            : points(points), coefficients(Eigen::ArrayXd::Zero(_pointsPerElement)) { }

        virtual ~Basis() { }

    public:
        // geometry definition
        static const unsigned pointsPerEdge = _pointsPerEdge;
        static const unsigned pointsPerElement = _pointsPerElement;

        // member
        Eigen::ArrayXXd const points;
        Eigen::ArrayXd coefficients;
    };

    // linear basis class definition
    class Linear : public Basis<2, 3> {
    public:
        // constructor
        Linear(Eigen::Ref<Eigen::ArrayXXd const> points, unsigned const one);

        // evaluation
        double evaluate(Eigen::Ref<Eigen::ArrayXd const> point) const;

        // mathematical evaluation of basis
        double integralA(Linear const& other) const;
        double integralB(Linear const& other) const;
        static double boundaryIntegral(Eigen::Ref<Eigen::ArrayXd const> const points,
            unsigned const one);

        // mesh related stuff
        static inline unsigned pointCount(std::shared_ptr<numeric::IrregularMesh const> const mesh)
            { return mesh->nodes.rows(); }
        static inline auto elementConnections(std::shared_ptr<numeric::IrregularMesh const> const mesh) ->
            decltype(mesh->elements) { return mesh->elements; }
        static inline auto edgeConnections(std::shared_ptr<numeric::IrregularMesh const> const mesh) ->
            decltype(mesh->edges) { return mesh->edges; }
        static inline unsigned toLocalIndex(std::shared_ptr<numeric::IrregularMesh const> const, unsigned const, unsigned const index)
            { return index; }
    };

    // quadratic basis class definition
    class Quadratic : public Basis<3, 6> {
    public:
        // constructor
        Quadratic(Eigen::Ref<Eigen::ArrayXXd const> points, unsigned const one);

        // evaluation
        double evaluate(Eigen::Ref<Eigen::ArrayXd const> point) const;

        // mathematical evaluation of basis
        double integralA(Quadratic const& other) const;
        double integralB(Quadratic const& other) const;
        static double boundaryIntegral(Eigen::Ref<Eigen::ArrayXd const> const points,
            unsigned const one);

        // mesh related stuff
        static inline unsigned pointCount(std::shared_ptr<numeric::IrregularMesh const> const mesh)
            { return mesh->nodes.rows(); }
        static inline auto elementConnections(std::shared_ptr<numeric::IrregularMesh const> const mesh) ->
            decltype(mesh->elements) { return mesh->elements; }
        static inline auto edgeConnections(std::shared_ptr<numeric::IrregularMesh const> const mesh) ->
            decltype(mesh->edges) { return mesh->edges; }
        static inline unsigned toLocalIndex(std::shared_ptr<numeric::IrregularMesh const> const, unsigned const, unsigned const index)
            { return index; }
    };

    // edge bases basis function definition
    class Edge : public Basis<1, 3> {
    public:
        // constructor
        Edge(Eigen::Ref<Eigen::ArrayXXd const> const points,
            Eigen::Ref<Eigen::ArrayXi const> const edge);

        // mathematical evaluation of basis
        double integralA(Edge const& other) const;
        double integralB(Edge const& other) const;
        static double boundaryIntegral(Eigen::Ref<Eigen::ArrayXd const> const points,
            unsigned const) { return std::abs(points(points.rows() - 1) - points(0)); }

        // mesh related stuff
        static inline unsigned pointCount(std::shared_ptr<numeric::IrregularMesh const> const mesh)
            { return mesh->edges.rows(); }
        static inline auto elementConnections(std::shared_ptr<numeric::IrregularMesh const> const mesh) ->
            decltype(mesh->elementEdges) { return mesh->elementEdges; }
        static inline std::function<unsigned(unsigned, unsigned)> edgeConnections(std::shared_ptr<numeric::IrregularMesh const> const)
            { return [](unsigned edge, unsigned) { return edge; }; }

        static Eigen::ArrayXi toLocalIndex(std::shared_ptr<numeric::IrregularMesh const> const mesh, unsigned const element, unsigned const index) {
            Eigen::ArrayXi const edge = mesh->edges.row(mesh->elementEdges(element, index));

            Eigen::ArrayXi localEdge(edge.rows());
            for (int node = 0; node < edge.rows(); ++node) {
                int nodeIndex = 0;
                (mesh->elements.row(element) - edge(node)).square().minCoeff(&nodeIndex);

                localEdge(node) = nodeIndex;
            }

            return localEdge;
        }

        // member
    protected:
        std::array<Linear, 2> nodeBasis;
        double length;
    };
}
}
}

#endif
