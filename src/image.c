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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <linalgcl/linalgcl.h>
#include "basis.h"
#include "mesh.h"
#include "image.h"

// create image program
linalgcl_error_t ert_image_program_create(ert_image_program_t* programPointer,
    cl_context context, cl_device_id device_id, const char* path) {
    // check input
    if ((programPointer == NULL) || (context == NULL) || (path == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;
    linalgcl_error_t linalgcl_error = LINALGCL_SUCCESS;

    // init program pointer
    *programPointer = NULL;

    // create program struct
    ert_image_program_t program = malloc(sizeof(ert_image_program_s));

    // check success
    if (program == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    program->program = NULL;
    program->kernel_calc_image_phi = NULL;
    program->kernel_calc_image_sigma = NULL;

    // read program file
    // open file
    FILE* file = fopen(path, "r");

    // check success
    if (file == NULL) {
        // cleanup
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    // get file length
    size_t length = 0;
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    fseek(file, 0, SEEK_SET);

    // allocate buffer
    char* buffer = malloc(sizeof(char) * length);

    // check success
    if (buffer == NULL) {
        // cleanup
        fclose(file);
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    // fread file
    if (fread(buffer, sizeof(char), length, file) != length) {
        // cleanup
        free(buffer);
        fclose(file);
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    // close file
    fclose(file);

    // create program from source buffer
    program->program = clCreateProgramWithSource(context, 1,
        (const char**)&buffer, &length, &cl_error);
    free(buffer);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    // build program
    cl_error = clBuildProgram(program->program, 0, NULL, NULL, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        // print build error log
        char buffer[2048];
        clGetProgramBuildInfo(program->program, device_id, CL_PROGRAM_BUILD_LOG,
            sizeof(buffer), buffer, NULL);
        printf("%s\n", buffer);

        // cleanup
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    // create kernel
    program->kernel_calc_image_phi = clCreateKernel(program->program,
        "calc_image_phi", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    program->kernel_calc_image_sigma = clCreateKernel(program->program,
        "calc_image_sigma", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_image_program_release(&program);

        return LINALGCL_ERROR;
    }

    // set program pointer
    *programPointer = program;

    return LINALGCL_SUCCESS;
}

// release image program
linalgcl_error_t ert_image_program_release(ert_image_program_t* programPointer) {
    // check input
    if ((programPointer == NULL) || (*programPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get program
    ert_image_program_t program = *programPointer;

    if (program->program != NULL) {
        clReleaseProgram(program->program);
    }

    if (program->kernel_calc_image_phi != NULL) {
        clReleaseKernel(program->kernel_calc_image_phi);
    }

    if (program->kernel_calc_image_sigma != NULL) {
        clReleaseKernel(program->kernel_calc_image_sigma);
    }

    // free struct
    free(program);

    // set program pointer to NULL
    *programPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create image
linalgcl_error_t ert_image_create(ert_image_t* imagePointer, linalgcl_size_t size_x,
    linalgcl_size_t size_y, ert_mesh_t mesh, cl_context context, cl_device_id device_id) {
    // check input
    if ((imagePointer == NULL) || (mesh == NULL) || (context == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init image pointer
    *imagePointer = NULL;

    // create image struct
    ert_image_t image = malloc(sizeof(ert_image_s));

    // check success
    if (image == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    image->elements = NULL;
    image->image = NULL;
    image->program = NULL;
    image->mesh = mesh;

    // create matrices
    error  = linalgcl_matrix_create(&image->elements, context,
        mesh->element_count, 18);
    error += linalgcl_matrix_create(&image->image, context,
        size_x, size_y);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_image_release(&image);

        return error;
    }

    // fill elements
    for (linalgcl_size_t i = 0; i < image->image->size_x; i++) {
        for (linalgcl_size_t j = 0; j < image->image->size_y; j++) {
            image->image->host_data[(i * image->image->size_y) + j] = NAN;
        }
    }

    // create program
    error = ert_image_program_create(&image->program, context, device_id, "src/image.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_image_release(&image);

        return error;
    }

    // fill elements matrix
    linalgcl_matrix_data_t x[3], y[3], id[3];
    ert_basis_t basis[3];

    for (linalgcl_size_t k = 0; k < mesh->element_count;k++) {
        // get vertices for element
        for (linalgcl_size_t i = 0; i < 3; i++) {
            linalgcl_matrix_get_element(mesh->elements, &id[i], k, i);
            linalgcl_matrix_get_element(mesh->vertices, &x[i], (linalgcl_size_t)id[i], 0);
            linalgcl_matrix_get_element(mesh->vertices, &y[i], (linalgcl_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        ert_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        ert_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        ert_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // set matrix elements
        for (linalgcl_size_t i = 0; i < 3; i++) {
            // ids
            linalgcl_matrix_set_element(image->elements, id[i], k, i);

            // coordinates
            linalgcl_matrix_set_element(image->elements, x[i], k, 3 + 2 * i);
            linalgcl_matrix_set_element(image->elements, y[i], k, 4 + 2 * i);

            // basis coefficients
            linalgcl_matrix_set_element(image->elements, basis[i]->coefficients[0], k, 9 + 3 * i);
            linalgcl_matrix_set_element(image->elements, basis[i]->coefficients[1], k, 10 + 3 * i);
            linalgcl_matrix_set_element(image->elements, basis[i]->coefficients[2], k, 11 + 3 * i);
        }

        // cleanup
        ert_basis_release(&basis[0]);
        ert_basis_release(&basis[1]);
        ert_basis_release(&basis[2]);
    }

    // set image pointer
    *imagePointer = image;

    return LINALGCL_SUCCESS;
}

// release image
linalgcl_error_t ert_image_release(ert_image_t* imagePointer) {
    // check input
    if ((imagePointer == NULL) || (*imagePointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get image
    ert_image_t image = *imagePointer;

    // cleanup
    linalgcl_matrix_release(&image->elements);
    linalgcl_matrix_release(&image->image);
    ert_image_program_release(&image->program);

    // free struct
    free(image);

    // set image pointer to NULL
    *imagePointer = NULL;

    return LINALGCL_SUCCESS;
}

// calc image
linalgcl_error_t ert_image_calc_phi(ert_image_t image,
    linalgcl_matrix_t phi, cl_command_queue queue) {
    // check input
    if ((image == NULL) || (phi == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(image->program->kernel_calc_image_phi,
        0, sizeof(cl_mem), &image->image->device_data);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_phi,
        1, sizeof(cl_mem), &image->elements->device_data);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_phi,
        2, sizeof(cl_mem), &phi->device_data);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_phi,
        3, sizeof(linalgcl_size_t), &image->image->size_x);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_phi,
        4, sizeof(linalgcl_size_t), &image->image->size_y);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_phi,
        5, sizeof(linalgcl_matrix_data_t), &image->mesh->radius);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = image->elements->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, image->program->kernel_calc_image_phi,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// calc image
linalgcl_error_t ert_image_calc_sigma(ert_image_t image,
    linalgcl_matrix_t sigma, cl_command_queue queue) {
    // check input
    if ((image == NULL) || (sigma == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(image->program->kernel_calc_image_sigma,
        0, sizeof(cl_mem), &image->image->device_data);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_sigma,
        1, sizeof(cl_mem), &image->elements->device_data);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_sigma,
        2, sizeof(cl_mem), &sigma->device_data);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_sigma,
        3, sizeof(linalgcl_size_t), &image->image->size_x);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_sigma,
        4, sizeof(linalgcl_size_t), &image->image->size_y);
    cl_error += clSetKernelArg(image->program->kernel_calc_image_sigma,
        5, sizeof(linalgcl_matrix_data_t), &image->mesh->radius);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = image->elements->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, image->program->kernel_calc_image_sigma,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

