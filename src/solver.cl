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

__kernel void calc_jacobian(__global float* jacobian, __global float* applied_phi,
    __global float* lead_phi, __global float* gradient_matrix_values,
    __global float* gradient_matrix_column_ids, __global float* area,
    unsigned int phi_size, unsigned int jacobian_size) {
    // get id
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    // calc matrix element
    float element = 0.0f;
    float2 grad_applied_phi = {0.0f, 0.0f};
    float2 grad_lead_phi = {0.0f, 0.0f};
    unsigned int id = 0;

    for (unsigned int k = 0; k < BLOCK_SIZE; k++) {
        id = gradient_matrix_column_ids[2 * j * BLOCK_SIZE + k];

        grad_applied_phi.x += gradient_matrix_values[2 * j * BLOCK_SIZE + id] * applied_phi[id * phi_size + 0];
        grad_applied_phi.y += gradient_matrix_values[(2 * j + 1) * BLOCK_SIZE + id] * applied_phi[id * phi_size + 0];
        grad_lead_phi.x += gradient_matrix_values[2 * j * BLOCK_SIZE + id] * lead_phi[id * phi_size + i];
        grad_lead_phi.y += gradient_matrix_values[(2 * j + 1) * BLOCK_SIZE + id] * lead_phi[id * phi_size + i];
    }
    element = -area[j] * (grad_applied_phi.x * grad_lead_phi.x + grad_applied_phi.y * grad_lead_phi.y);

    // set matrix element
    jacobian[(i * jacobian_size) + j] = element;
}
