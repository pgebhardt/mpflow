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
Eigen::ArrayXd mpFlow::math::polar(Eigen::Ref<Eigen::ArrayXd const> const point) {
    // calc radius
    double angle = 0.0;
    double radius = sqrt(point.square().sum());

    // calc angle
    if (point(0) > 0.0) {
        angle = atan(point(1) / point(0));
    }
    else if ((point(0) < 0.0) && (point(1) >= 0.0)) {
        angle = atan(point(1) / point(0)) + M_PI;
    }
    else if ((point(0) < 0.0) && (point(1) < 0.0)) {
        angle = atan(point(1) / point(0)) - M_PI;
    }
    else if ((point(0) == 0.0) && (point(1) > 0.0)) {
        angle = M_PI / 2.0;
    }
    else if ((point(0) == 0.0) && (point(1) < 0.0)) {
        angle = - M_PI / 2.0;
    }
    else {
        angle = 0.0;
    }

    Eigen::ArrayXd result(2);
    result << radius, angle;
    return result;
}

// convert polar to kartesian coordinates
Eigen::ArrayXd mpFlow::math::kartesian(Eigen::Ref<Eigen::ArrayXd const> const point) {
    Eigen::ArrayXd result(2);
    result(0) = point(0) * cos(point(1));
    result(1) = point(0) * sin(point(1));

    return result;
}

// calc circle parameter
double mpFlow::math::circleParameter(Eigen::Ref<Eigen::ArrayXd const> const point,
    double const offset) {
    // convert to polar coordinates
    Eigen::ArrayXd polarPoint = polar(point);
    double angle = polarPoint(1);

    // correct angle
    angle -= offset / polarPoint(0);
    angle += (angle < -M_PI) ? 2.0 * M_PI : 0.0;
    angle -= (angle > M_PI) ? 2.0 * M_PI : 0.0;

    // calc parameter
    return angle * polarPoint(0);
}

Eigen::ArrayXd mpFlow::math::arange(double const start, double const end, double const distance) {
    // calculate size of output array
    unsigned const count = ceil((end - start) / distance);
    
    Eigen::ArrayXd result = Eigen::ArrayXd::Zero(count);
    for (unsigned i = 0; i < count; ++i) {
        result(i) = start + (double)i * distance;
    }

    return result;
}

Eigen::ArrayXXd mpFlow::math::circularPoints(double const radius, double const distance,
    double const offset, bool const invertDirection, Eigen::Ref<Eigen::ArrayXd const> const midpoint) {
    Eigen::ArrayXd const phi = arange(offset / radius, offset / radius + 2.0 * M_PI, distance / radius);

    Eigen::ArrayXXd result = Eigen::ArrayXXd::Zero(phi.rows(), 2);
    result.col(0) = midpoint(0) + radius * phi.cos();
    result.col(1) = midpoint(1) + (invertDirection ? -1.0 : 1.0) * radius * phi.sin();

    return result;
}