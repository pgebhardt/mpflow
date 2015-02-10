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

#ifndef MPFLOW_INCLUDE_MATH_H
#define MPFLOW_INCLUDE_MATH_H

// namespace mpFlow::math
namespace mpFlow {
namespace math {
    // square
    template <
        class type
    >
    inline type square(type value) { return value * value; }

    // convert kartesian to polar coordinates
    std::tuple<dtype::real, dtype::real> polar(std::tuple<dtype::real, dtype::real> point);

    // convert polar to kartesian coordinates
    std::tuple<dtype::real, dtype::real> kartesian(std::tuple<dtype::real, dtype::real> point);

    // calc circle parameter
    dtype::real circleParameter(std::tuple<dtype::real, dtype::real> point,
        dtype::real offset);

    // round to size
    inline dtype::size roundTo(dtype::size size, dtype::size block_size) {
        return size == 0 ? 0 : (size / block_size + 1) * block_size;
    }

    // simple gauss elemination
    Eigen::ArrayXf gaussElemination(Eigen::ArrayXXf matrix,
        Eigen::ArrayXf excitation);
}
}

#endif
