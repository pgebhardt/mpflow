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

#ifndef FASTECT_IMAGE_H
#define FASTECT_IMAGE_H

// fastect image
typedef struct {
    linalgcu_matrix_t elements;
    linalgcu_matrix_t image;
    fastect_mesh_t mesh;
} fastect_image_s;
typedef fastect_image_s* fastect_image_t;

// create image
linalgcu_error_t fastect_image_create(fastect_image_t* imagePointer, linalgcu_size_t size_x,
    linalgcu_size_t size_y, fastect_mesh_t mesh, cudaStream_t stream);

// release image
linalgcu_error_t fastect_image_release(fastect_image_t* imagePointer);

// calc image
LINALGCU_EXTERN_C
linalgcu_error_t fastect_image_calc_phi(fastect_image_t image, linalgcu_matrix_t phi,
    cudaStream_t stream);

// calc image
LINALGCU_EXTERN_C
linalgcu_error_t fastect_image_calc_sigma(fastect_image_t image, linalgcu_matrix_t sigma,
    cudaStream_t stream);

#endif
