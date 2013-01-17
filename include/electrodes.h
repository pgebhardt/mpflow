// fastEIT
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_ELECTRODES_H
#define FASTEIT_INCLUDE_ELECTRODES_H

// namespace fastEIT
namespace fastEIT {
    // Electrodes class definition
    template <
        class mesh_type
    >
    class Electrodes {
    // constructer and destructor
    public:
        Electrodes(dtype::size count, std::tuple<dtype::real, dtype::real> shape,
            dtype::real impedance, const std::shared_ptr<mesh_type> mesh);
        virtual ~Electrodes() { }

    public:
        // accessor
        dtype::size count() const { return this->count_; }
        const std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>>& coordinates() const {
            return this->coordinates_;
        }
        const std::tuple<dtype::real, dtype::real> shape() const { return this->shape_; }
        dtype::real impedance() const { return this->impedance_; }

    // member
    private:
        dtype::size count_;
        std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>> coordinates_;
        std::tuple<dtype::real, dtype::real> shape_;
        dtype::real impedance_;
    };
}

#endif
