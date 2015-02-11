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

#ifndef MPFLOW_INCLUDE_NUMERIC_IRREGULAR_MESH_H
#define MPFLOW_INCLUDE_NUMERIC_IRREGULAR_MESH_H

// namespace mpFlow::numeric
namespace mpFlow {
namespace numeric {
    // class for holdingding irregular meshs
    class IrregularMesh {
    public:
        // constructor
        IrregularMesh(std::shared_ptr<Matrix<dtype::real>> nodes,
            std::shared_ptr<Matrix<dtype::index>> elements,
            std::shared_ptr<Matrix<dtype::index>> boundary, dtype::real radius,
            dtype::real height);

        // helper methods
        std::tuple<Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1>,
            Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>> elementNodes(dtype::index element);
        std::tuple<Eigen::Array<mpFlow::dtype::index, Eigen::Dynamic, 1>,
            Eigen::Array<mpFlow::dtype::real, Eigen::Dynamic, Eigen::Dynamic>> boundaryNodes(dtype::index element);

        // member
        std::shared_ptr<Matrix<dtype::real>> nodes;
        std::shared_ptr<Matrix<dtype::index>> elements;
        std::shared_ptr<Matrix<dtype::index>> boundary;
        dtype::real radius;
        dtype::real height;
    };

    // mesh helper
    namespace irregularMesh {
        // create mesh for quadratic basis function
        std::shared_ptr<mpFlow::numeric::IrregularMesh> quadraticBasis(
            std::shared_ptr<Matrix<dtype::real>> nodes,
            std::shared_ptr<Matrix<dtype::index>> elements,
            std::shared_ptr<Matrix<dtype::index>> boundary,
            dtype::real radius, dtype::real height, cudaStream_t stream);

        // quadratic mesh from linear
        std::tuple<
            std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>>,
            std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>,
            std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>>> quadraticMeshFromLinear(
            const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::real>> nodes_old,
            const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements_old,
            const std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> boundary_old,
            cudaStream_t stream);

        std::tuple<
            Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>,
            Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>,
            Eigen::Array<dtype::index, Eigen::Dynamic, Eigen::Dynamic>>
            calculateGlobalEdgeIndices(std::shared_ptr<mpFlow::numeric::Matrix<mpFlow::dtype::index>> elements);
    }
}
}

#endif
