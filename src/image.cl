#define BLOCK_SIZE (16)

__kernel void calc_image(__global float* image, __global float* elements, __global float* phi,
    unsigned int image_size_x, unsigned int image_size_y) {
    // get id
    unsigned int k = get_global_id(0);

    // get element data
    unsigned int id[3];
    float x[3], y[3], basis[3][3];

    for (int i = 0; i < 3; i++) {
        // ids
        id[i] = elements[(k * 2 * BLOCK_SIZE) + i];

        // coordinates
        x[i] = elements[(k * 2 * BLOCK_SIZE) + 3 + 2 * i];
        y[i] = elements[(k * 2 * BLOCK_SIZE) + 4 + 2 * i];

        // basis coefficients
        basis[i][0] = elements[(k * 2 * BLOCK_SIZE) + 9 + 3 * i];
        basis[i][1] = elements[(k * 2 * BLOCK_SIZE) + 10 + 3 * i];
        basis[i][2] = elements[(k * 2 * BLOCK_SIZE) + 11 + 3 * i];
    }
}
