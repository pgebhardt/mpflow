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
        class Source {
        public:
            // constructor
            Source(std::string type, dtype::real value, std::shared_ptr<Mesh> mesh,
                std::shared_ptr<Electrodes> electrodes, dtype::size components_count,
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
            virtual void initCEM(cublasHandle_t, cudaStream_t) {
            };

        public:
            // accessors
            std::string& type() { return this->type_; }
            std::shared_ptr<Mesh> mesh() { return this->mesh_; }
            std::shared_ptr<Electrodes> electrodes() { return this->electrodes_; }
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
            std::shared_ptr<Matrix<dtype::real>> d_matrix() {
                return this->d_matrix_;
            }
            std::shared_ptr<Matrix<dtype::real>> w_matrix() {
                return this->w_matrix_;
            }
            std::shared_ptr<Matrix<dtype::real>> x_matrix() {
                return this->x_matrix_;
            }
            std::shared_ptr<Matrix<dtype::real>> z_matrix() {
                return this->z_matrix_;
            }
            std::shared_ptr<Matrix<dtype::real>> excitation(dtype::index index) {
                return this->excitation_[index];
            }
            dtype::size drive_count() { return this->drive_pattern()->columns(); }
            dtype::size measurement_count() { return this->measurement_pattern()->columns(); }
            dtype::real value() { return this->value_; }
            dtype::size components_count() { return this->components_count_; }

        private:
            // member
            std::string type_;
            std::shared_ptr<Mesh> mesh_;
            std::shared_ptr<Electrodes> electrodes_;
            std::shared_ptr<Matrix<dtype::real>> drive_pattern_;
            std::shared_ptr<Matrix<dtype::real>> measurement_pattern_;
            std::shared_ptr<Matrix<dtype::real>> pattern_;
            std::array<std::shared_ptr<Matrix<dtype::real>>, 2> elemental_pattern_;
            std::shared_ptr<Matrix<dtype::real>> d_matrix_;
            std::shared_ptr<Matrix<dtype::real>> w_matrix_;
            std::shared_ptr<Matrix<dtype::real>> x_matrix_;
            std::shared_ptr<Matrix<dtype::real>> z_matrix_;
            std::vector<std::shared_ptr<Matrix<dtype::real>>> excitation_;
            dtype::real value_;
            dtype::size components_count_;
        };

        // current source
        template <
            class basis_function_type
        >
        class Current : public Source {
        public:
            // constructor
            Current(dtype::real current, std::shared_ptr<Mesh> mesh,
                std::shared_ptr<Electrodes> electrodes, dtype::size components_count,
                std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
                cublasHandle_t handle, cudaStream_t stream);

            // update excitation
            virtual void updateExcitation(cublasHandle_t handle, cudaStream_t stream);

        protected:
            // init excitation
            virtual void initCEM(cublasHandle_t handle, cudaStream_t stream);
        };

        // voltage source
        template <
            class basis_function_type
        >
        class Voltage : public Source {
        public:
            // constructor
            Voltage(dtype::real voltage, std::shared_ptr<Mesh> mesh,
                std::shared_ptr<Electrodes> electrodes, dtype::size components_count,
                std::shared_ptr<Matrix<dtype::real>> drive_pattern,
                std::shared_ptr<Matrix<dtype::real>> measurement_pattern,
                cublasHandle_t handle, cudaStream_t stream);

            // update excitation
            virtual void updateExcitation(cublasHandle_t handle, cudaStream_t stream);

        protected:
            // init excitation
            virtual void initCEM(cublasHandle_t handle, cudaStream_t stream);
        };
    }
}

#endif
