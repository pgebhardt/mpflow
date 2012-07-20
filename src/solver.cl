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
    unsigned int jacobian_size, unsigned int measurment_count,
    unsigned int drive_count) {
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
    unsigned int idx, idy;

    for (unsigned int k = 0; k < 3; k++) {
        idx = (unsigned int)gradient_matrix_column_ids[2 * j * BLOCK_SIZE + k];

        if ((idx == 0) && (k != 0)) {
            break;
        }

        // x gradient
        grad_applied_phi.x -= gradient_matrix_values[2 * j * BLOCK_SIZE + k] * applied_phi[idx * drive_count + drive_id];
        grad_lead_phi.x -= gradient_matrix_values[2 * j * BLOCK_SIZE + k] * lead_phi[idx * measurment_count + measurment_id];
    }

    for (unsigned int k = 0; k < 3; k++) {
        idy = (unsigned int)gradient_matrix_column_ids[(2 * j + 1) * BLOCK_SIZE + k];

        if ((idy == 0) && (k != 0)) {
            break;
        }

        // y gradient
        grad_applied_phi.y -= gradient_matrix_values[(2 * j + 1) * BLOCK_SIZE + k] * applied_phi[idy * drive_count + drive_id];
        grad_lead_phi.y -= gradient_matrix_values[(2 * j + 1) * BLOCK_SIZE + k] * lead_phi[idy * measurment_count + measurment_id];
    }
    element = area[j] * (grad_applied_phi.x * grad_lead_phi.x + grad_applied_phi.y * grad_lead_phi.y);

    // set matrix element
    jacobian[(i * jacobian_size) + j] = element;
}

__kernel void calc_gradient(__global float* gradient, __global float* jacobian,
    __global float* measured_voltage, __global float* calculated_voltage, __global float* sigma,
    unsigned int size_x, unsigned int size_y) {
    // get id
    unsigned int i = get_global_id(0);

    // calc element
    float element = 0.0f;
    for (unsigned int j = 0; j < size_x; j++) {
        element += 2.0f * (measured_voltage[(j % 16) * 32 + j / 16] - calculated_voltage[(j % 16) * 32 + j / 16]) * jacobian[j * size_y + i]
            - (1.0f - sigma[i]) * (1.0f - sigma[i]);
    }

    // set element
    gradient[i] = -element;
}
