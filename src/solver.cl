__kernel void update_system_matrix(__global float* temp) {
    // get id
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(0);

    temp[i + j] = 0.0;
}
