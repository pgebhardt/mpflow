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

#ifndef MPFLOW_INCLUDE_NUMERIC_IRREGULAR_MESH_H
#define MPFLOW_INCLUDE_NUMERIC_IRREGULAR_MESH_H

namespace mpFlow {
namespace numeric {
    // class for holdingding irregular meshs
    class IrregularMesh {
    public:
        // constructor
        IrregularMesh(Eigen::Ref<Eigen::ArrayXXd const> const nodes,
            Eigen::Ref<Eigen::ArrayXXi const> const elements,
            Eigen::Ref<Eigen::ArrayXXi const> const edges);
        IrregularMesh(Eigen::Ref<Eigen::ArrayXXd const> const nodes,
            Eigen::Ref<Eigen::ArrayXXi const> const elements);

        // factories
        static std::shared_ptr<IrregularMesh> fromConfig(json_value const& config,
            json_value const& portConfig, cudaStream_t const stream, std::string const path="./");

        // helper methods
        Eigen::ArrayXXd elementNodes(unsigned const element) const;
        Eigen::ArrayXXd edgeNodes(unsigned const edge) const;

        // interpolation
        template <
            class interpolationFunctionType,
            class dataType
        >
        std::shared_ptr<numeric::SparseMatrix<dataType>> createInterpolationMatrix(
            std::shared_ptr<IrregularMesh const> const mesh, cudaStream_t const stream) const;

        // member
        Eigen::ArrayXXd const nodes;
        Eigen::ArrayXXi const elements;
        Eigen::ArrayXXi const edges;
        Eigen::ArrayXXi const elementEdges;
        Eigen::ArrayXi const boundary;
    };
/*
    // mesh helper
    namespace irregularMesh {
        // create mesh for quadratic basis function
        std::shared_ptr<mpFlow::numeric::IrregularMesh> quadraticBasis(
            Eigen::Ref<Eigen::ArrayXXd const> const nodes, Eigen::Ref<Eigen::ArrayXXi const> const elements,
            Eigen::Ref<Eigen::ArrayXXi const> const boundary, double const height);

        // quadratic mesh from linear
        std::tuple<Eigen::ArrayXXd, Eigen::ArrayXXi, Eigen::ArrayXXi> quadraticMeshFromLinear(
            Eigen::Ref<Eigen::ArrayXXd const> const nodes, Eigen::Ref<Eigen::ArrayXXi const> const elements,
            Eigen::Ref<Eigen::ArrayXXi const> const boundary);
    }
*/
}
}

#endif
