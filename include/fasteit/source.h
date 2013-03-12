// fastEIT
//
// Copyright (C) 2013  Patrik Gebhardt
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
        public:
            // constructor
            Source(std::string type, dtype::real value, std::shared_ptr<model_type> model,
                std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
                cublasHandle_t handle, cudaStream_t stream);

            // destructor
            virtual ~Source() { }

            // update excitation
            virtual void updateExcitation(cublasHandle_t, cudaStream_t) {
            };

        protected:
            // init excitation
            void initExcitation(cublasHandle_t handle, cudaStream_t stream);

        public:
            // accessors
            std::string& type() { return this->type_; }
            std::shared_ptr<model_type> model() { return this->model_; }
            std::shared_ptr<Matrix<dtype::real>> drive_pattern() {
                return this->drive_pattern_;
            }
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern() {
                return this->measurement_pattern_;
            }
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
            dtype::size drive_count() { return this->drive_pattern()->columns(); }
            dtype::size measurement_count() { return this->measurement_pattern()->columns(); }
            dtype::real value() { return this->value_; }

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
