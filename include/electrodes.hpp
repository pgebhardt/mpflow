// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_ELECTRODES_HPP
#define FASTEIT_INCLUDE_ELECTRODES_HPP

// Electrodes class definition
class Electrodes {
// constructer and destructor
public:
    Electrodes(dtype::size count, dtype::real width, dtype::real height, dtype::real meshRadius);
    virtual ~Electrodes();

// accessor
public:
    dtype::size count() const { return this->count_; }
    inline const std::tuple<dtype::real, dtype::real> electrodes_start(dtype::index index) const {
        assert(index < this->count());
        return std::make_tuple(this->electrodes_start_[index * 2 + 0], this->electrodes_start_[index * 2 + 1]);
    }
    inline const std::tuple<dtype::real, dtype::real> electrodes_end(dtype::index index) const {
        assert(index < this->count());
        return std::make_tuple(this->electrodes_end_[index * 2 + 0], this->electrodes_end_[index * 2 + 1]);
    }
    dtype::real width() const { return this->width_; }
    dtype::real height() const { return this->height_; }

// member
private:
    dtype::size count_;
    dtype::real* electrodes_start_;
    dtype::real* electrodes_end_;
    dtype::real width_;
    dtype::real height_;
};

#endif
