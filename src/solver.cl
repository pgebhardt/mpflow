#define BLOCK_SIZE (16)

__kernel void copy_to_column(__global float* matrix, __global float* vector, unsigned int column,
    unsigned int size_y) {
    // get id
    unsigned int i = get_global_id(0);

    matrix[(i * size_y) + column] = vector[i];
}

__kernel void copy_from_column(__global float* matrix, __global float* vector, unsigned int column,
    unsigned int size_y) {
    // get id
    unsigned int i = get_global_id(0);

    vector[i] = matrix[(i * size_y) + column];
}
