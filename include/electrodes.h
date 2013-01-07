// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_ELECTRODES_H
#define FASTEIT_INCLUDE_ELECTRODES_H

// namespace fastEIT
namespace fastEIT {
    // Electrodes class definition
    template <
        class MeshType
    >
    class Electrodes {
    // constructer and destructor
    public:
        Electrodes(dtype::size count, dtype::real width, dtype::real height,
            const std::shared_ptr<MeshType> mesh,
            std::shared_ptr<Matrix<dtype::real>> drive_pattern,
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern);
        virtual ~Electrodes() { }

    public:
        // accessor
        dtype::size count() const { return this->count_; }
        const std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>>& coordinates() const {
            return this->coordinates_;
        }
        dtype::real width() const { return this->width_; }
        dtype::real height() const { return this->height_; }
        std::shared_ptr<Matrix<dtype::real>> drive_pattern() const { return this->drive_pattern_; }
        std::shared_ptr<Matrix<dtype::real>> measurement_pattern() const { return this->measurement_pattern_; }
        dtype::size drive_count() const { return this->drive_pattern()->columns(); }
        dtype::size measurement_count() const { return this->measurement_pattern()->columns(); }

    // member
    private:
        dtype::size count_;
        std::vector<std::tuple<std::tuple<dtype::real, dtype::real>,
            std::tuple<dtype::real, dtype::real>>> coordinates_;
        dtype::real width_;
        dtype::real height_;
        std::shared_ptr<Matrix<dtype::real>> drive_pattern_;
        std::shared_ptr<Matrix<dtype::real>> measurement_pattern_;
    };
}

#endif
