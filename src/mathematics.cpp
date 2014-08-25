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

#include "mpflow/mpflow.h"

// convert kartesian to polar coordinates
std::tuple<mpFlow::dtype::real, mpFlow::dtype::real>
    mpFlow::math::polar(std::tuple<dtype::real, dtype::real> point) {
    // calc radius
    dtype::real angle = 0.0f;
    dtype::real radius = sqrt(square(std::get<0>(point)) + square(std::get<1>(point)));

    // calc angle
    if (std::get<0>(point) > 0.0f) {
        angle = atan(std::get<1>(point) / std::get<0>(point));
    }
    else if ((std::get<0>(point) < 0.0f) && (std::get<1>(point) >= 0.0f)) {
        angle = atan(std::get<1>(point) / std::get<0>(point)) + M_PI;
    }
    else if ((std::get<0>(point) < 0.0f) && (std::get<1>(point) < 0.0f)) {
        angle = atan(std::get<1>(point) / std::get<0>(point)) - M_PI;
    }
    else if ((std::get<0>(point) == 0.0f) && (std::get<1>(point) > 0.0f)) {
        angle = M_PI / 2.0f;
    }
    else if ((std::get<0>(point) == 0.0f) && (std::get<1>(point) < 0.0f)) {
        angle = - M_PI / 2.0f;
    }
    else {
        angle = 0.0f;
    }

    return std::make_tuple(radius, angle);
}

// convert polar to kartesian coordinates
std::tuple<mpFlow::dtype::real, mpFlow::dtype::real>
    mpFlow::math::kartesian(std::tuple<dtype::real, dtype::real> point) {
    dtype::real x = std::get<0>(point) * cos(std::get<1>(point));
    dtype::real y = std::get<0>(point) * sin(std::get<1>(point));

    return std::make_tuple(x, y);
}

// calc circle parameter
mpFlow::dtype::real mpFlow::math::circleParameter(
    std::tuple<dtype::real, dtype::real> point, dtype::real offset) {
    // convert to polar coordinates
    std::tuple<dtype::real, dtype::real> polar_point = polar(point);
    dtype::real angle = std::get<1>(polar_point);

    // correct angle
    angle -= offset / std::get<0>(polar_point);
    angle += (angle < -M_PI) ? 2.0f * M_PI : 0.0f;
    angle -= (angle > M_PI) ? 2.0f * M_PI : 0.0f;

    // calc parameter
    return angle * std::get<0>(polar_point);
}
