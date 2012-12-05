// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_ELECTRODES_H
#define FASTEIT_INCLUDE_ELECTRODES_H

// namespace fastEIT
namespace fastEIT {
    // Electrodes class definition
    class Electrodes {
    // constructer and destructor
    public:
        Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real meshRadius);
        virtual ~Electrodes() { }

    public:
        // accessor
        dtype::size count() const { return this->count_; }
        const std::vector<std::tuple<dtype::real, dtype::real> >& electrodes_start() const {
            return this->electrodes_start_;
        }
        const std::vector<std::tuple<dtype::real, dtype::real> >& electrodes_end() const {
            return this->electrodes_end_;
        }
        dtype::real width() const { return this->width_; }
        dtype::real height() const { return this->height_; }

    // member
    private:
        dtype::size count_;
        std::vector<std::tuple<dtype::real, dtype::real> > electrodes_start_;
        std::vector<std::tuple<dtype::real, dtype::real> > electrodes_end_;
        dtype::real width_;
        dtype::real height_;
    };
}

#endif
