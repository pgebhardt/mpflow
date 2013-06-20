// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_EIT_ELECTRODES_H
#define MPFLOW_INCLUDE_EIT_ELECTRODES_H

// namespace mpFlow
namespace mpFlow {
namespace EIT {
    // Electrodes class definition
    class Electrodes {
    // constructer and destructor
    public:
        Electrodes(dtype::size count, std::tuple<dtype::real, dtype::real> shape,
            dtype::real impedance);
        virtual ~Electrodes() { }

    public:
        // accessor
        dtype::size count() const { return this->count_; }
        const std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>& coordinates(dtype::index index) const {
            return this->coordinates_[index];
        }
        std::tuple<dtype::real, dtype::real> shape() const { return this->shape_; }
        dtype::real area() const { return std::get<0>(this->shape()) * std::get<1>(this->shape()); }
        dtype::real impedance() const { return this->impedance_; }

        // mutators
        std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>& coordinates(dtype::index index) {
            return this->coordinates_[index];
        }

    // member
    private:
        dtype::size count_;
        std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>> coordinates_;
        std::tuple<dtype::real, dtype::real> shape_;
        dtype::real impedance_;
    };

    // electrodes helper
    namespace electrodes {
        // create electrodes on circular boundary
        std::shared_ptr<Electrodes> circularBoundary(dtype::size count,
            std::tuple<dtype::real, dtype::real> shape, dtype::real impedance,
            dtype::real boundary_radius);
    }
}
}

#endif
