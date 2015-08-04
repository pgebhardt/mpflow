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

#ifndef MPFLOW_INCLUDE_FEM_PORTS_H
#define MPFLOW_INCLUDE_FEM_PORTS_H

namespace mpFlow {
namespace FEM {
    class Ports {
    public:
        Ports(Eigen::Ref<Eigen::ArrayXXi const> const edges,
            double const height);
        virtual ~Ports() { }

        // factories
        static std::shared_ptr<Ports> circularBoundary(unsigned const count,
            double const width, double const height, std::shared_ptr<numeric::IrregularMesh const> const mesh,
            double const offset=0.0, bool const clockwise=false);
        static std::shared_ptr<Ports> fromConfig(json_value const& config,
            std::shared_ptr<numeric::IrregularMesh const> const mesh);

        // member
        Eigen::ArrayXXi const edges;
        double const height;
        unsigned const count;
    };
}
}

#endif
