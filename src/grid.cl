#define BLOCK_SIZE (16)

__kernel void update_system_matrix(__global float* system_matrix_values,
    __global float* system_matrix_column_ids,
    __global float* gradient_matrix_transposed_values,
    __global float* gradient_matrix_transposed_column_ids,
    __global float* gradient_matrix_transposed,
    __global float* sigma,
    __global float* area,
    unsigned int element_count) {
    // get ids
    unsigned int i = get_global_id(0);
    unsigned int j = system_matrix_column_ids[(i * BLOCK_SIZE) + get_global_id(1)];

    // calc system_matrix_element
    float element = 0.0f;
    unsigned int id = 0;

    for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
        // get id
        id = gradient_matrix_transposed_column_ids[(i * BLOCK_SIZE) + k];

        element += gradient_matrix_transposed_values[(i * BLOCK_SIZE) + k] *
            sigma[id / 2] * area[id / 2] *
            gradient_matrix_transposed[(j * element_count) + id];
    }

    system_matrix_values[(i * BLOCK_SIZE) + get_global_id(1)] = (((j == 0) && (get_global_id(1) == 0)) ||
        (j != 0)) ? element : 0.0f;
}
