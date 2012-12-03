#ifndef FASTEIT_MATH_HPP
#define FASTEIT_MATH_HPP

// namespace fastEIT
namespace fastEIT {
    // namespace math
    namespace math {
        // square
        template <class type>
        static inline type square(type value) { return value * value; }

        // convert kartesian to polar coordinates
        static void polar(dtype::real& radius, dtype::real& angle, dtype::real x, dtype::real y) {
            // calc radius
            radius = sqrt(square(x) + square(y));

            // calc angle
            if (x > 0.0f) {
                angle = atan(y / x);
            }
            else if ((x < 0.0f) && (y >= 0.0f)) {
                angle = atan(y / x) + M_PI;
            }
            else if ((x < 0.0f) && (y < 0.0f)) {
                angle = atan(y / x) - M_PI;
            }
            else if ((x == 0.0f) && (y > 0.0f)) {
                angle = M_PI / 2.0f;
            }
            else if ((x == 0.0f) && (y < 0.0f)) {
                angle = - M_PI / 2.0f;
            }
            else {
                angle = 0.0f;
            }
        }

        // convert polar to kartesian coordinates
        static void kartesian(dtype::real& x, dtype::real& y, dtype::real radius, dtype::real angle) {
            x = radius * cos(angle);
            y = radius * sin(angle);
        }

        // calc circle parameter
        static dtype::real circleParameter(std::tuple<dtype::real, dtype::real> point, dtype::real offset) {
            // convert to polar coordinates
            dtype::real radius, angle;
            polar(radius, angle, std::get<0>(point), std::get<1>(point));

            // correct angle
            angle -= offset / radius;
            angle += (angle < M_PI) ? 2.0f * M_PI : 0.0f;
            angle -= (angle > M_PI) ? 2.0f * M_PI : 0.0f;

            // calc parameter
            return angle * radius;
        }
    }
}

#endif
