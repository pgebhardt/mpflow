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

// ert image program
typedef struct {
    cl_program program;
    cl_kernel kernel_calc_image_phi;
    cl_kernel kernel_calc_image_sigma;
} ert_image_program_s;
typedef ert_image_program_s* ert_image_program_t;

// ert image
typedef struct {
    linalgcl_matrix_t elements;
    linalgcl_matrix_t image;
    ert_image_program_t program;
    ert_mesh_t mesh;
} ert_image_s;
typedef ert_image_s* ert_image_t;

// create image program
linalgcl_error_t ert_image_program_create(ert_image_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path);

// release image program
linalgcl_error_t ert_image_program_release(ert_image_program_t* programPointer);

// create image
linalgcl_error_t ert_image_create(ert_image_t* imagePointer, linalgcl_size_t size_x,
    linalgcl_size_t size_y, ert_mesh_t mesh, cl_context context, cl_device_id device_id);

// release image
linalgcl_error_t ert_image_release(ert_image_t* imagePointer);

// calc image
linalgcl_error_t ert_image_calc_phi(ert_image_t image, linalgcl_matrix_t phi,
    cl_command_queue queue);

// calc image
linalgcl_error_t ert_image_calc_sigma(ert_image_t image, linalgcl_matrix_t sigma,
    cl_command_queue queue);

#endif
