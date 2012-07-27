// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

// redefine extern c
#define LINALGCU_EXTERN_C extern "C"

#include <stdlib.h>
#include "fastect.h"

// helper functions
__device__ linalgcu_matrix_data_t test(
    linalgcu_matrix_data_t ax, linalgcu_matrix_data_t ay,
    linalgcu_matrix_data_t bx, linalgcu_matrix_data_t by,
    linalgcu_matrix_data_t cx, linalgcu_matrix_data_t cy) {
    return (ax - cx) * (by - cy) - (bx - cx) * (ay - cy);
}

__device__ linalgcu_bool_t pointInTriangle(
    linalgcu_matrix_data_t px, linalgcu_matrix_data_t py,
    linalgcu_matrix_data_t ax, linalgcu_matrix_data_t ay,
    linalgcu_matrix_data_t bx, linalgcu_matrix_data_t by,
    linalgcu_matrix_data_t cx, linalgcu_matrix_data_t cy) {
    linalgcu_bool_t b1, b2, b3;

    b1 = test(px, py, ax, ay, bx, by) <= 0.00000045f;
    b2 = test(px, py, bx, by, cx, cy) <= 0.00000045f;
    b3 = test(px, py, cx, cy, ax, ay) <= 0.00000045f;

    return ((b1 == b2) && (b2 == b3));
}

// clac image phi kernel
__global__ void calc_image_phi_kernel(linalgcu_matrix_data_t* image,
    linalgcu_matrix_data_t* elements, linalgcu_matrix_data_t* phi,
    linalgcu_size_t size_x, linalgcu_size_t size_y, linalgcu_matrix_data_t radius) {
    // get id
    linalgcu_size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    // get element data
    linalgcu_column_id_t id[3];
    linalgcu_matrix_data_t xVertex[3], yVertex[3], basis[3][3];

    for (int i = 0; i < 3; i++) {
        // ids
        id[i] = elements[k * 2 * LINALGCU_BLOCK_SIZE + i];

        // coordinates
        xVertex[i] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 3 + 2 * i];
        yVertex[i] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 4 + 2 * i];

        // basis coefficients
        basis[i][0] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 9 + 3 * i];
        basis[i][1] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 10 + 3 * i];
        basis[i][2] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 11 + 3 * i];
    }

    // step size
    float dx = 2.0f * radius / ((float)size_x - 1.0f);
    float dy = 2.0f * radius / ((float)size_y - 1.0f);

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
    float pixel = 0.0f;
    float x, y;
    for (int i = iStart; i <= iEnd; i++) {
        for (int j = jStart; j <= jEnd; j++) {
            // calc coordinate
            x = (float)i * dx - radius;
            y = (float)j * dy - radius;

            // calc pixel
            pixel  = phi[id[0]] * (basis[0][0] + basis[0][1] * x + basis[0][2] * y);
            pixel += phi[id[1]] * (basis[1][0] + basis[1][1] * x + basis[1][2] * y);
            pixel += phi[id[2]] * (basis[2][0] + basis[2][1] * x + basis[2][2] * y);

            // set pixel
            if (pointInTriangle(x, y, xVertex[0], yVertex[0], xVertex[1], yVertex[1],
                    xVertex[2], yVertex[2])) {
                image[i + j * size_x] = pixel;
            }
        }
    }
}

// calc image phi
extern "C"
linalgcu_error_t fastect_image_calc_phi(fastect_image_t image,
    linalgcu_matrix_t phi, cudaStream_t stream) {
    // check input
    if ((image == NULL) || (phi == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    calc_image_phi_kernel<<<image->elements->size_n / LINALGCU_BLOCK_SIZE,
        LINALGCU_BLOCK_SIZE, 0, stream>>>(image->image->device_data,
        image->elements->device_data, phi->device_data, image->image->size_m,
        image->image->size_n, image->mesh->radius);

    return LINALGCU_SUCCESS;
}

// clac image phi kernel
__global__ void calc_image_sigma_kernel(linalgcu_matrix_data_t* image,
    linalgcu_matrix_data_t* elements, linalgcu_matrix_data_t* sigma,
    linalgcu_size_t size_x, linalgcu_size_t size_y, linalgcu_matrix_data_t radius) {
    // get id
    linalgcu_size_t k = blockIdx.x * blockDim.x + threadIdx.x;

    // get element data
    linalgcu_matrix_data_t xVertex[3], yVertex[3];

    for (int i = 0; i < 3; i++) {
        // coordinates
        xVertex[i] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 3 + 2 * i];
        yVertex[i] = elements[k * 2 * LINALGCU_BLOCK_SIZE + 4 + 2 * i];
    }

    // step size
    float dx = 2.0f * radius / ((float)size_x - 1.0f);
    float dy = 2.0f * radius / ((float)size_y - 1.0f);

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
    float pixel = 0.0f;
    float x, y;
    for (int i = iStart; i <= iEnd; i++) {
        for (int j = jStart; j <= jEnd; j++) {
            // calc coordinate
            x = (float)i * dx - radius;
            y = (float)j * dy - radius;

            // calc pixel
            pixel = sigma[k];

            // set pixel
            if (pointInTriangle(x, y, xVertex[0], yVertex[0], xVertex[1], yVertex[1],
                    xVertex[2], yVertex[2])) {
                image[i + j * size_x] = pixel;
            }
        }
    }
}

// calc image phi
extern "C"
linalgcu_error_t fastect_image_calc_sigma(fastect_image_t image,
    linalgcu_matrix_t sigma, cudaStream_t stream) {
    // check input
    if ((image == NULL) || (sigma == NULL)) {
        return LINALGCU_ERROR;
    }

    // execute kernel
    calc_image_sigma_kernel<<<image->elements->size_n / LINALGCU_BLOCK_SIZE,
        LINALGCU_BLOCK_SIZE, 0, stream>>>(image->image->device_data,
        image->elements->device_data, sigma->device_data, image->image->size_m,
        image->image->size_n, image->mesh->radius);

    return LINALGCU_SUCCESS;
}
