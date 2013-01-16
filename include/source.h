// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_SOURCE_H
#define FASTEIT_INCLUDE_SOURCE_H

// namespace fastEIT
namespace fastEIT {
    // source namespace
    namespace source {
        // source base class
        class Source {
        protected:
            // constructor
            Source(std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern);

        public:
            // destructor
            virtual ~Source() { }

            // accessors
            const std::shared_ptr<Matrix<dtype::real>> drive_pattern() const {
                return this->drive_pattern_;
            }
            const std::shared_ptr<Matrix<dtype::real>> measurement_pattern() const {
                return this->measurement_pattern_;
            }
            dtype::size drive_count() const { return this->drive_pattern()->columns(); }
            dtype::size measurement_count() const { return this->measurement_pattern()->columns(); }

        private:
            // member
            std::shared_ptr<Matrix<dtype::real>> drive_pattern_;
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern_;
        };

        // current source
        class Current : public Source {
        public:
            // constructor
            Current(dtype::real current, std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern);

            // accessors
            dtype::real current() const { return this->current_; }

        private:
            // member
            dtype::real current_;
        };

        // voltage source
        class Voltage : public Source {
        public:
            // constructor
            Voltage(dtype::real voltage, std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern);

            // accessors
            dtype::real voltage() const { return this->voltage_; }

        private:
            // member
            dtype::real voltage_;
        };
    }
}

#endif
