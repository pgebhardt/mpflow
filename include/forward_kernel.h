// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FORWARD_KERNEL_H
#define FASTEIT_INCLUDE_FORWARD_KERNEL_H

// namespace fastEIT
namespace fastEIT {
    // namespace forward
    namespace forwardKernel {
        // calc jacobian kernel
        template <
            int nodes_per_element
        >
        void calcJacobian(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* drive_phi, const dtype::real* measurment_phi,
            const dtype::index* connectivity_matrix,
            const dtype::real* elemental_jacobian_matrix, const dtype::real* gamma,
            dtype::real sigma_ref, dtype::size rows, dtype::size columns,
            dtype::size phi_rows, dtype::size element_count, dtype::size drive_count,
            dtype::size measurment_count, bool additiv, dtype::real* jacobian);

        // apply boundary condition
        void applyBoundaryCondition(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* boundary, dtype::real* values, dtype::index* column_ids,
            dtype::size density);
    }
}

#endif
