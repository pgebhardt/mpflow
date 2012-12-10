// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FORWARD_CUDA_H
#define FASTEIT_INCLUDE_FORWARD_CUDA_H

// namespace fastEIT
namespace fastEIT {
    // namespace forward
    namespace forward {
        // calc jacobian
        template <
            int nodes_per_element
        >
        void calcJacobian(const Matrix<dtype::real>* gamma, const Matrix<dtype::real>* phi,
            const Matrix<dtype::index>* elements, const Matrix<dtype::real>* elemental_jacobian_matrix,
            dtype::size drive_count, dtype::size measurment_count, dtype::real sigma_ref, bool additiv,
            cudaStream_t stream, Matrix<dtype::real>* jacobian);
    }
}

#endif
