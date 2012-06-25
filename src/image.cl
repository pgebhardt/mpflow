
#define BLOCK_SIZE (16)

float test(float ax, float ay, float bx, float by, float cx, float cy) {
    return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy);
}

bool pointInTriangle(float px, float py, float ax, float ay, float bx, float by,
    float cx, float cy) {
    bool b1, b2, b3;

    b1 = test(px, py, ax, ay, bx, by) <= 0.0001;
    b2 = test(px, py, bx, by, cx, cy) <= 0.0001;
    b3 = test(px, py, cx, cy, ax, ay) <= 0.0001;

    return ((b1 == b2) && (b2 == b3));
}

__kernel void calc_image(__global float* image, __global float* elements, __global float* phi,
    unsigned int size_x, unsigned int size_y) {
    // get id
    unsigned int k = get_global_id(0);

    // get element data
    unsigned int id[3];
    float xVertex[3], yVertex[3], basis[3][3];

    for (int i = 0; i < 3; i++) {
        // ids
        id[i] = elements[(k * 2 * BLOCK_SIZE) + i];

        // coordinates
        xVertex[i] = elements[(k * 2 * BLOCK_SIZE) + 3 + 2 * i];
        yVertex[i] = elements[(k * 2 * BLOCK_SIZE) + 4 + 2 * i];

        // basis coefficients
        basis[i][0] = elements[(k * 2 * BLOCK_SIZE) + 9 + 3 * i];
        basis[i][1] = elements[(k * 2 * BLOCK_SIZE) + 10 + 3 * i];
        basis[i][2] = elements[(k * 2 * BLOCK_SIZE) + 11 + 3 * i];
    }

    // step size
    float dx = 2.0 / ((float)size_x - 1.0);
    float dy = 2.0 / ((float)size_y - 1.0);

    // start and stop indices
    int iStart = (int)(min(min(xVertex[0], xVertex[1]),
        xVertex[2]) / dx) + size_x / 2;
    int jStart = (int)(min(min(yVertex[0], yVertex[1]),
        yVertex[2]) / dy) + size_y / 2;
    int iEnd = (int)(max(max(xVertex[0], xVertex[1]),
        xVertex[2]) / dx) + size_x / 2;
    int jEnd = (int)(max(max(yVertex[0], yVertex[1]),
        yVertex[2]) / dy) + size_y / 2;

    // calc triangle
    float pixel = 0.0;
    float x, y;
    for (unsigned int i = iStart; i <= iEnd; i++) {
        for (unsigned int j = jStart; j <= jEnd; j++) {
            // calc coordinate
            x = (float)i * dx - 1.0;
            y = (float)j * dy - 1.0;

            // calc pixel
            pixel  = phi[id[0]] * (basis[0][0] + basis[0][1] * x + basis[0][2] * y);
            pixel += phi[id[1]] * (basis[1][0] + basis[1][1] * x + basis[1][2] * y);
            pixel += phi[id[2]] * (basis[2][0] + basis[2][1] * x + basis[2][2] * y);

            // set pixel
            if (pointInTriangle(x, y, xVertex[0], yVertex[0], xVertex[1], yVertex[1],
                    xVertex[2], yVertex[2])) {
                image[(i * size_y) + j] = pixel;
            }
        }
    }
}

