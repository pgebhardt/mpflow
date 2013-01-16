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
            Source(dtype::size drive_count, dtype::size measurement_count)
                : drive_count_(drive_count), measurement_count_(measurement_count) { }

        public:
            // destructor
            virtual ~Source() { }

            // accessors
            dtype::size drive_count() const { return this->drive_count_; }
            dtype::size measurement_count() const { return this->measurement_count_; }

        private:
            // member
            dtype::size drive_count_;
            dtype::size measurement_count_;
        };

        // current source
        class Current : public Source {
        public:
            // constructor
            Current(dtype::real current, std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern);

            // accessors
            dtype::real current() const { return this->current_; }
            const std::shared_ptr<Matrix<dtype::real>> drive_pattern() const {
                return this->drive_pattern_;
            }
            const std::shared_ptr<Matrix<dtype::real>> measurement_pattern() {
                return this->measurement_pattern_;
            }

        private:
            // member
            dtype::real current_;
            std::shared_ptr<Matrix<dtype::real>> drive_pattern_;
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern_;
        };

        // voltage source
        class Voltage : public Source {
        public:
            // constructor
            Voltage(dtype::size drive_count, dtype::size measurement_count)
                : Source(drive_count, measurement_count) { }
        };
    }
}

#endif
