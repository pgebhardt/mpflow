// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_MATH_H
#define MPFLOW_INCLUDE_MATH_H

// namespace mpFlow
namespace mpFlow {
    // namespace math
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
        template <
            class type,
            mpFlow::dtype::size size
        >
        inline std::array<type, size> gaussElemination(
            std::array<std::array<type, size>, size> matrix,
            std::array<type, size> excitation) {
            dtype::real x, sum;
            dtype::index n = size;

            // foraward elemination
            for (mpFlow::dtype::size k = 0; k < n - 1; ++k) {
                // find index of maximum pivot
                auto pivot_index = k;
                for (mpFlow::dtype::size i = k; i < n; ++i) {
                    pivot_index = std::abs(matrix[i][k]) > std::abs(matrix[pivot_index][k]) ?
                        i : pivot_index;
                }

                // swap rows
                for (mpFlow::dtype::index i = 0; i < n; ++i) {
                    std::tie(matrix[pivot_index][i], matrix[k][i]) =
                        std::make_tuple(matrix[k][i], matrix[pivot_index][i]);
                }
                std::tie(excitation[pivot_index], excitation[k]) =
                    std::make_tuple(excitation[k], excitation[pivot_index]);

                for (mpFlow::dtype::size i = k + 1; i < n; ++i) {
                    x = matrix[i][k] / matrix[k][k];

                    for (mpFlow::dtype::size j = k; j < n; ++j) {
                        matrix[i][j] = matrix[i][j] - matrix[k][j] * x;
                    }
                    excitation[i] = excitation[i] - excitation[k] * x;
                }
            }

            // Resubstitution
            excitation[n - 1] = excitation[n - 1] / matrix[n - 1][n - 1];
            for (mpFlow::dtype::size i = n - 2;; --i) {
                sum = excitation[i];

                for (mpFlow::dtype::size j = i + 1; j < n; ++j) {
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
