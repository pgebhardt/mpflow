// mpFlow
//
// Copyright (C) 2014  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_UWB_WINDOWS_H
#define MPFLOW_INCLUDE_UWB_WINDOWS_H

// namespace mpFlow
namespace mpFlow {
namespace UWB {
    // Windows class definition
    class Windows {
    // constructer and destructor
    public:
        Windows(dtype::size count, std::tuple<dtype::real, dtype::real> shape);
        virtual ~Windows() { }

    public:
        // accessor
        dtype::size count() const { return this->_count; }
        const std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>& coordinates(dtype::index index) const {
            return this->_coordinates[index];
        }
        std::tuple<dtype::real, dtype::real> shape() const { return this->_shape; }
        dtype::real area() const { return std::get<0>(this->shape()) * std::get<1>(this->shape()); }

        // mutators
        std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>& coordinates(dtype::index index) {
            return this->_coordinates[index];
        }

    // member
    private:
        dtype::size _count;
        std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>> _coordinates;
        std::tuple<dtype::real, dtype::real> _shape;
    };

    // windows helper
    namespace windows {
        // create electrodes on circular boundary
        std::shared_ptr<Windows> circularBoundary(dtype::size count,
            std::tuple<dtype::real, dtype::real> shape, dtype::real boundary_radius);
    }
}
}

#endif
