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
        std::array<type, size> gaussElemination(
            std::array<std::array<type, size>, size> matrix,
            std::array<type, size> excitation);
    }
}

#endif
