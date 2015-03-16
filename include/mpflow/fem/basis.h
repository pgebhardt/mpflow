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
        Basis(Eigen::Ref<const Eigen::ArrayXXd> points)
            : points(points), coefficients(Eigen::ArrayXd::Zero(_pointsPerElement)) { }

        virtual ~Basis() { }

    public:
        // geometry definition
        static const unsigned pointsPerEdge = _pointsPerEdge;
        static const unsigned pointsPerElement = _pointsPerElement;

        // member
        Eigen::ArrayXXd points;
        Eigen::ArrayXd coefficients;
    };

    // linear basis class definition
    class Linear : public Basis<2, 3> {
    public:
        // constructor
        Linear(Eigen::Ref<const Eigen::ArrayXXd> points, unsigned one);

        // mathematical evaluation of basis
        double integrateWithBasis(const std::shared_ptr<Linear> other);
        double integrateGradientWithBasis(const std::shared_ptr<Linear> other);
        static double integrateBoundaryEdge(Eigen::Ref<const Eigen::ArrayXd> points,
            unsigned one, double start, double end);
    };

    // quadratic basis class definition
    class Quadratic : public Basis<3, 6> {
    public:
        // constructor
        Quadratic(Eigen::Ref<const Eigen::ArrayXXd> points, unsigned one);

        // mathematical evaluation of basis
        double integrateWithBasis(const std::shared_ptr<Quadratic> other);
        double integrateGradientWithBasis(const std::shared_ptr<Quadratic> other);
        static double integrateBoundaryEdge(Eigen::Ref<const Eigen::ArrayXd> points,
            unsigned one, double start, double end);
    };

    // edge bases basis function definition
    class Edge : public Basis<1, 3> {
    public:
        // constructor
        Edge(Eigen::Ref<const Eigen::ArrayXXd> points,
            Eigen::Ref<const Eigen::ArrayXi> edge);

        // mathematical evaluation of basis
        double integrateWithBasis(const std::shared_ptr<Edge> other);
        double integrateGradientWithBasis(const std::shared_ptr<Edge> other);

        // member
        std::array<Linear, 2> nodeBasis;
        double length;
    };
}
}
}

#endif
