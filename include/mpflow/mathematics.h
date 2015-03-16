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
    std::tuple<double, double> polar(std::tuple<double, double> point);

    // convert polar to kartesian coordinates
    std::tuple<double, double> kartesian(std::tuple<double, double> point);

    // calc circle parameter
    double circleParameter(std::tuple<double, double> point,
        double offset);

    // round to size
    inline unsigned roundTo(unsigned size, unsigned block_size) {
        return size == 0 ? 0 : (size / block_size + 1) * block_size;
    }

    // simple gauss elemination
    template <
        class type,
        unsigned size
    >
    inline std::array<type, size> gaussElemination(
        std::array<std::array<type, size>, size> matrix,
        std::array<type, size> excitation) {
        double x, sum;
        unsigned n = size;

        // foraward elemination
        for (unsigned k = 0; k < n - 1; ++k) {
            // find index of maximum pivot
            auto pivot_index = k;
            for (unsigned i = k; i < n; ++i) {
                pivot_index = std::abs(matrix[i][k]) > std::abs(matrix[pivot_index][k]) ?
                    i : pivot_index;
            }

            // swap rows
            for (unsigned i = 0; i < n; ++i) {
                std::tie(matrix[pivot_index][i], matrix[k][i]) =
                    std::make_tuple(matrix[k][i], matrix[pivot_index][i]);
            }
            std::tie(excitation[pivot_index], excitation[k]) =
                std::make_tuple(excitation[k], excitation[pivot_index]);

            for (unsigned i = k + 1; i < n; ++i) {
                x = matrix[i][k] / matrix[k][k];

                for (unsigned j = k; j < n; ++j) {
                    matrix[i][j] = matrix[i][j] - matrix[k][j] * x;
                }
                excitation[i] = excitation[i] - excitation[k] * x;
            }
        }

        // Resubstitution
        excitation[n - 1] = excitation[n - 1] / matrix[n - 1][n - 1];
        for (unsigned i = n - 2;; --i) {
            sum = excitation[i];

            for (unsigned j = i + 1; j < n; ++j) {
                sum = sum - matrix[i][j] * excitation[j];
            }

            excitation[i] = sum / matrix[i][i];

            // break condition
            if (i == 0) {
                break;
            }
        }

        return excitation;
    }
}
}

#endif
