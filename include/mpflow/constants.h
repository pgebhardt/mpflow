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

#ifndef MPFLOW_INCLUDE_CONSTANTS_H
#define MPFLOW_INCLUDE_CONSTANTS_H

namespace mpFlow {
namespace constants {
    // electromagnetic wave
    const double epsilon0 = 8.8541878176e-12;
    const double mu0 = 1.2566370614e-6;
    const double c0 = 1.0 / std::sqrt(epsilon0 * mu0);
    static const unsigned invalid_index = (unsigned)(-1);
}
}

#endif
