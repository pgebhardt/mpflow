#define BLOCK_SIZE (16)

__kernel void regularize_system_matrix(__global float* result, __global float* system_matrix,
    unsigned int size_y, float sigma) {
    // get id
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    // calc matrix element
    float element = 0.0f;
    for (unsigned int k = 0; k < size_y; k++) {
        element += system_matrix[(i * size_y) + k] * system_matrix[(k * size_y) + j];
    }

    // set result
    result[(i * size_y) + j] = (i == j) ? element + sigma : element;
}

__kernel void update_vector(__global float* result, __global float* x1, float sign,
    __global float* x2, __global float* r1, __global float* r2) {
    // get id
    unsigned int i = get_global_id(0);

    // calc value
    result[i] = x1[i] + sign * x2[i] * r1[0] / r2[0];
}
