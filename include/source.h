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
        template <
            class model_type
        >
        class Source {
        protected:
            // constructor
            Source(std::string type, dtype::real value, std::shared_ptr<model_type> model,
                std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
                cublasHandle_t handle, cudaStream_t stream);

        public:
            // destructor
            virtual ~Source() { }

            // update excitation
            virtual void updateExcitation(cublasHandle_t handle, cudaStream_t stream) = 0;

        protected:
            // init excitation
            void initExcitation(cublasHandle_t handle, cudaStream_t stream);

        public:
            // accessors
            const std::string& type() const { return this->type_; }
            const std::shared_ptr<model_type> model() const { return this->model_; }
            const std::shared_ptr<Matrix<dtype::real>> drive_pattern() const {
                return this->drive_pattern_;
            }
            const std::shared_ptr<Matrix<dtype::real>> measurement_pattern() const {
                return this->measurement_pattern_;
            }
            const std::shared_ptr<Matrix<dtype::real>> pattern() const {
                return this->pattern_;
            }
            const std::shared_ptr<Matrix<dtype::real>> elemental_pattern(dtype::index index) const {
                return this->elemental_pattern_[index];
            }
            const std::shared_ptr<Matrix<dtype::real>> excitation_matrix() const {
                return this->excitation_matrix_;
            }
            const std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) const {
                return this->excitation_[index];
            }
            dtype::size drive_count() const { return this->drive_pattern()->columns(); }
            dtype::size measurement_count() const { return this->measurement_pattern()->columns(); }
            dtype::real value() const { return this->value_; }

            // mutators
            std::shared_ptr<model_type> model() { return this->model_; }
            std::shared_ptr<Matrix<dtype::real>> pattern() {
                return this->pattern_;
            }
            std::shared_ptr<Matrix<dtype::real>> elemental_pattern(dtype::index index) {
                return this->elemental_pattern_[index];
            }
            std::shared_ptr<Matrix<dtype::real>> excitation_matrix() {
                return this->excitation_matrix_;
            }
            std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) {
                return this->excitation_[index];
            }

        private:
            // member
            std::string type_;
            std::shared_ptr<model_type> model_;
            std::shared_ptr<Matrix<dtype::real>> drive_pattern_;
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern_;
            std::shared_ptr<Matrix<dtype::real>> pattern_;
            std::array<std::shared_ptr<Matrix<dtype::real>>, 2> elemental_pattern_;
            std::shared_ptr<Matrix<dtype::real>> excitation_matrix_;
            std::vector<std::shared_ptr<Matrix<dtype::real>>> excitation_;
            dtype::real value_;
        };

        // current source
        template <
            class model_type
        >
        class Current : public Source<model_type> {
        public:
            // constructor
            Current(dtype::real current, std::shared_ptr<model_type> model,
                std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
                cublasHandle_t handle, cudaStream_t stream);

            // update excitation
            virtual void updateExcitation(cublasHandle_t handle, cudaStream_t stream);
        };

        // voltage source
        template <
            class model_type
        >
        class Voltage : public Source<model_type> {
        public:
            // constructor
            Voltage(dtype::real voltage, std::shared_ptr<model_type> model,
                std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
                cublasHandle_t handle, cudaStream_t stream);

            // update excitation
            virtual void updateExcitation(cublasHandle_t handle, cudaStream_t stream);
        };
    }
}

#endif
