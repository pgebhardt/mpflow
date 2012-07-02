#define BLOCK_SIZE (16)

__kernel void add_scalar(__global float* vector, __global float* scalar) {
    // get id
    unsigned int i = get_global_id(0);

    vector[i] += scalar[0];
}

__kernel void update_vector(__global float* result, __global float* x1, float sign,
    __global float* x2, __global float* r1, __global float* r2) {
    // get id
    unsigned int i = get_global_id(0);

    // calc value
    result[i] = x1[i] + sign * x2[i] * r1[0] / r2[0];
}
