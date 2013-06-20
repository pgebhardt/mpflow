// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLDUE_EIT_MODEL_KERNEL_H
#define MPFLOW_INCLDUE_EIT_MODEL_KERNEL_H

// namespaces mpFlow::EIT::modelKernel
namespace mpFlow {
namespace EIT {
namespace modelKernel {
    // reduce connectivity and elemental matrices
    template <
        class type
    >
    void reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const type* intermediate_matrix, const dtype::index* column_ids,
        dtype::size rows, dtype::index offset, type* matrix);

    // update matrix kernel
    void updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dtype::index* connectivity_matrix, const dtype::real* elemental_matrix,
        const dtype::real* gamma, dtype::real sigma_ref, dtype::size rows,
        dtype::size columns, dtype::real* matrix_values);

    // update system matrix kernel
    void updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dtype::real* s_matrix_values, const dtype::real* r_matrix_values,
        const dtype::index* s_matrix_column_ids, const dtype::real* z_matrix,
        dtype::size density, dtype::real scalar, dtype::size z_matrix_rows,
        dtype::real* system_matrix_values);

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
}
}
}

#endif
