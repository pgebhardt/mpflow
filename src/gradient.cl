
#define BLOCK_SIZE (16)

__kernel void regularize_system_matrix(__global float* system_matrix_values, __global float* system_matrix_column_ids,
    float sigma) {
    // get id
    unsigned int i = get_global_id(0);

    for (unsigned int j = 0; j < BLOCK_SIZE; j++) {
        // get column_id
        system_matrix_values[(i * BLOCK_SIZE) + j] += (float)i == system_matrix_column_ids[(i * BLOCK_SIZE) + j] ? sigma : 0.0;
    }
}

