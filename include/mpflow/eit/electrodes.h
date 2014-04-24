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

#ifndef MPFLOW_INCLUDE_EIT_ELECTRODES_H
#define MPFLOW_INCLUDE_EIT_ELECTRODES_H

// namespace mpFlow
namespace mpFlow {
namespace EIT {
    class Electrodes {
    public:
        Electrodes(dtype::size count, std::tuple<dtype::real, dtype::real> shape);
        virtual ~Electrodes() { }

        // member
        dtype::size count;
        std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>> coordinates;
        std::tuple<dtype::real, dtype::real> shape;
    };

    // electrodes helper
    namespace electrodes {
        // create electrodes on circular boundary
        std::shared_ptr<Electrodes> circularBoundary(dtype::size count,
            std::tuple<dtype::real, dtype::real> shape, dtype::real boundaryRadius);
    }
}
}

#endif
