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

#ifndef MPFLOW_INCLUDE_FEM_BOUNDARY_DESCRIPTOR_H
#define MPFLOW_INCLUDE_FEM_BOUNDARY_DESCRIPTOR_H

namespace mpFlow {
namespace FEM {
    class BoundaryDescriptor {
    public:
        BoundaryDescriptor(dtype::size count, std::tuple<dtype::real, dtype::real> shape);
        virtual ~BoundaryDescriptor() { }

        // member
        dtype::size count;
        std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>> coordinates;
        std::tuple<dtype::real, dtype::real> shape;
    };

    namespace boundaryDescriptor {
        // create BoundaryDescriptor on circular boundary
        std::shared_ptr<BoundaryDescriptor> circularBoundary(dtype::size count,
            std::tuple<dtype::real, dtype::real> shape, dtype::real boundaryRadius,
            dtype::real offset);
    }
}
}

#endif
