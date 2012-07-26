// ert
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

#ifndef ERT_IMAGE_H
#define ERT_IMAGE_H

// ert image
typedef struct {
    linalgcu_matrix_t elements;
    linalgcu_matrix_t image;
    ert_mesh_t mesh;
} ert_image_s;
typedef ert_image_s* ert_image_t;

// create image
linalgcu_error_t ert_image_create(ert_image_t* imagePointer, linalgcu_size_t size_x,
    linalgcu_size_t size_y, ert_mesh_t mesh, cudaStream_t stream);

// release image
linalgcu_error_t ert_image_release(ert_image_t* imagePointer);

// calc image
LINALGCU_EXTERN_C
linalgcu_error_t ert_image_calc_phi(ert_image_t image, linalgcu_matrix_t phi,
    cudaStream_t stream);

// calc image
LINALGCU_EXTERN_C
linalgcu_error_t ert_image_calc_sigma(ert_image_t image, linalgcu_matrix_t sigma,
    cudaStream_t stream);

#endif
