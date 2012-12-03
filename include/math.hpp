#ifndef FASTEIT_INCLUDE_MATH_HPP
#define FASTEIT_INCLUDE_MATH_HPP

// namespace fastEIT
namespace fastEIT {
    // namespace math
    namespace math {
        // square
        template <class type>
        static inline type square(type value) { return value * value; }

        // convert kartesian to polar coordinates
        static std::tuple<dtype::real, dtype::real> polar(std::tuple<dtype::real, dtype::real> point) {
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
        static std::tuple<dtype::real, dtype::real> kartesian(std::tuple<dtype::real, dtype::real> point) {
            dtype::real x = std::get<0>(point) * cos(std::get<1>(point));
            dtype::real y = std::get<0>(point) * sin(std::get<1>(point));

            return std::make_tuple(x, y);
        }

        // calc circle parameter
        static dtype::real circleParameter(std::tuple<dtype::real, dtype::real> point, dtype::real offset) {
            // convert to polar coordinates
            std::tuple<dtype::real, dtype::real> polar_point = polar(point);
            dtype::real angle = std::get<1>(polar_point);

            // correct angle
            angle -= offset / std::get<0>(polar_point);
            angle += (angle < M_PI) ? 2.0f * M_PI : 0.0f;
            angle -= (angle > M_PI) ? 2.0f * M_PI : 0.0f;

            // calc parameter
            return angle * std::get<0>(polar_point);
        }
    }
}

#endif
