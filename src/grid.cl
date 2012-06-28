#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#define BLOCK_SIZE (16)

__kernel void update_system_matrix(__global double* system_matrix_values,
    __global double* system_matrix_column_ids,
    __global double* gradient_matrix_transposed_values,
    __global double* gradient_matrix_transposed_column_ids,
    __global double* gradient_matrix_transposed,
    __global double* sigma,
    __global double* area,
    unsigned int element_count) {
    // get ids
    unsigned int i = get_global_id(0);
    unsigned int j = system_matrix_column_ids[(i * BLOCK_SIZE) + get_global_id(1)];

    // calc system_matrix_element
    double element = 0.0;
    unsigned int id = 0;

    for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
        // get id
        id = gradient_matrix_transposed_column_ids[(i * BLOCK_SIZE) + k];

        element += gradient_matrix_transposed_values[(i * BLOCK_SIZE) + k] *
            sigma[id / 2] * area[id / 2] *
            gradient_matrix_transposed[(j * element_count) + id];
    }

    system_matrix_values[(i * BLOCK_SIZE) + get_global_id(1)] = (((j == 0) && (get_global_id(1) == 0)) ||
        (j != 0)) ? element : 0.0;
}

__kernel void unfold_system_matrix(__global double* result, __global double* system_matrix_values,
    __global double* system_matrix_column_ids, unsigned int size_y) {
    // get id
    unsigned int i = get_global_id(0);

    unsigned int column_id = 0;

    for (unsigned int j = 0; j < BLOCK_SIZE; j++) {
        // get column_id
        column_id = (unsigned int)system_matrix_column_ids[(i * BLOCK_SIZE) + j];

        result[(i * size_y) + column_id] += system_matrix_values[(i * BLOCK_SIZE) + j];
    }
}

__kernel void regulize_system_matrix(__global double* system_matrix,
    unsigned int size_y, double lambda) {
    // get id
    unsigned int i = get_global_id(0);

    system_matrix[(i * size_y) + i] -= lambda * lambda;
}

