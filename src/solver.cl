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
    unsigned int phi_size, unsigned int jacobian_size, unsigned int measurment_count) {
    // get id
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    // calc measurment and drive id
    unsigned int measurment_id = i % measurment_count;
    unsigned int drive_id = i / measurment_count;

    // calc matrix element
    float element = 0.0f;
    float2 grad_applied_phi = {0.0f, 0.0f};
    float2 grad_lead_phi = {0.0f, 0.0f};
    unsigned int id = 0;

    for (unsigned int k = 0; k < 3; k++) {
        id = gradient_matrix_column_ids[2 * j * BLOCK_SIZE + k];

        grad_applied_phi.x += gradient_matrix_values[2 * j * BLOCK_SIZE + id] * applied_phi[id * phi_size + drive_id];
        grad_applied_phi.y += gradient_matrix_values[(2 * j + 1) * BLOCK_SIZE + id] * applied_phi[id * phi_size + drive_id];
        grad_lead_phi.x += gradient_matrix_values[2 * j * BLOCK_SIZE + id] * lead_phi[id * phi_size + measurment_id];
        grad_lead_phi.y += gradient_matrix_values[(2 * j + 1) * BLOCK_SIZE + id] * lead_phi[id * phi_size + measurment_id];
    }
    element = -area[j] * (grad_applied_phi.x * grad_lead_phi.x + grad_applied_phi.y * grad_lead_phi.y);

    // set matrix element
    jacobian[(i * jacobian_size) + j] = element;
}

__kernel void regularize_jacobian(__global float* result, __global float* jacobian,
    float alpha, unsigned int size_x, unsigned int size_y) {
    // get id
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    // calc Jt * J
    float element = 0;
    for (unsigned int k = 0; k < size_x; k++) {
        element += jacobian[(k * size_y) + i] * jacobian[(k * size_y) + j];
    }

    // regularize
    element += i == j ? alpha : 0.0f;

    // set element
    result[(i * size_y) + j] = element;
}

__kernel void calc_sigma_excitation(__global float* result, __global float* jacobian,
    __global float* calculated_voltage, __global float* measured_voltage,
    __global float* initial_sigma, __global float* sigma, float alpha, unsigned int size_x,
    unsigned int size_y) {
    // get id
    unsigned int i = get_global_id(0);

    // calc sigma difference
    float dSigma = sigma[i] - initial_sigma[i];

    // calc element
    float element = 0.0f;
    for (unsigned int j = 0; j < size_x; j++) {
        element += jacobian[(i * size_y) + j] * (calculated_voltage[j] - measured_voltage[j]);
    }

    // regularize
    element -= alpha * (sigma[i] - initial_sigma[i]);

    // set element
    result[i] = element;
}
