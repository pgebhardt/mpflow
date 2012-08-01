// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

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
