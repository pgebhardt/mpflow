// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_MATH_H
#define FASTEIT_INCLUDE_MATH_H

// namespace fastEIT
namespace fastEIT {
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
            return (size / block_size + 1) * block_size;
        }

        // simple gauss elemination
        template <
            class type,
            fastEIT::dtype::size size
        >
        inline std::array<type, size> gaussElemination(
            std::array<std::array<type, size>, size> matrix,
            std::array<type, size> excitation) {
            dtype::real x, sum;
            dtype::index n = size;

            // foraward elemination
            for (fastEIT::dtype::size k = 0; k < n - 1; ++k) {
                for (fastEIT::dtype::size i = k + 1; i < n; ++i) {
                    x = matrix[i][k] / matrix[k][k];

                    for (fastEIT::dtype::size j = k + 1; j < n; ++j) {
                        matrix[i][j] = matrix[i][j] - matrix[k][j] * x;
                    }
                    excitation[i] = excitation[i] - excitation[k] * x;
                }
            }

            // Resubstitution
            excitation[n - 1] = excitation[n - 1] / matrix[n - 1][n - 1];
            for (fastEIT::dtype::size i = n - 2;; --i) {
                sum = excitation[i];

                for (fastEIT::dtype::size j = i + 1; j < n; ++j) {
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
