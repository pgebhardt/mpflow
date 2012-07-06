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
#include "electrodes.h"
#include "grid.h"

// create new grid program
linalgcl_error_t ert_grid_program_create(ert_grid_program_t* programPointer,
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
    ert_grid_program_t program = malloc(sizeof(ert_grid_program_s));

    // check success
    if (program == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    program->program = NULL;
    program->kernel_update_system_matrix = NULL;

    // read program file
    // open file
    FILE* file = fopen(path, "r");

    // check success
    if (file == NULL) {
        // cleanup
        ert_grid_program_release(&program);

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
        ert_grid_program_release(&program);

        return LINALGCL_ERROR;
    }

    // fread file
    if (fread(buffer, sizeof(char), length, file) != length) {
        // cleanup
        free(buffer);
        fclose(file);
        ert_grid_program_release(&program);

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
        ert_grid_program_release(&program);

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
        ert_grid_program_release(&program);

        return LINALGCL_ERROR;
    }

    // create kernel
    program->kernel_update_system_matrix = clCreateKernel(program->program,
        "update_system_matrix", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_grid_program_release(&program);

        return LINALGCL_ERROR;
    }

    // set program pointer
    *programPointer = program;

    return LINALGCL_SUCCESS;
}

// release solver program
linalgcl_error_t ert_grid_program_release(ert_grid_program_t* programPointer) {
    // check input
    if ((programPointer == NULL) || (*programPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get program
    ert_grid_program_t program = *programPointer;

    if (program->program != NULL) {
        clReleaseProgram(program->program);
    }

    if (program->kernel_update_system_matrix != NULL) {
        clReleaseKernel(program->kernel_update_system_matrix);
    }

    // free struct
    free(program);

    // set program pointer to NULL
    *programPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create solver grid
linalgcl_error_t ert_grid_create(ert_grid_t* gridPointer,
    linalgcl_matrix_program_t matrix_program, ert_mesh_t mesh,
    cl_context context, cl_device_id device_id, cl_command_queue queue) {
    // check input
    if ((gridPointer == NULL) || (mesh == NULL) ||
        (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init grid pointer
    *gridPointer = NULL;

    // create grid struct
    ert_grid_t grid = malloc(sizeof(ert_grid_s));

    // check success
    if (grid == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    grid->mesh = mesh;
    grid->system_matrix = NULL;
    grid->exitation_matrix = NULL;
    grid->gradient_matrix_sparse = NULL;
    grid->gradient_matrix = NULL;
    grid->sigma = NULL;
    grid->area = NULL;
    grid->program = NULL;

    // create matrices
    error  = linalgcl_matrix_create(&grid->gradient_matrix, context,
        grid->mesh->vertex_count, 2 * grid->mesh->element_count);
    error |= linalgcl_matrix_create(&grid->sigma, context,
        grid->mesh->element_count, 1);
    error |= linalgcl_matrix_create(&grid->area, context,
        grid->mesh->element_count, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // init to uniform sigma
    for (linalgcl_size_t i = 0; i < grid->mesh->element_count; i++) {
        linalgcl_matrix_set_element(grid->sigma, 1.0, i, 0);
    }

    // copy data to device
    linalgcl_matrix_copy_to_device(grid->gradient_matrix, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(grid->sigma, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(grid->area, queue, CL_FALSE);

    // create program
    error = ert_grid_program_create(&grid->program, context, device_id,
        "src/grid.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_grid_release(&grid);

        return error;
    }

    // init system matrix
    error = ert_grid_init_system_matrix(grid, matrix_program, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_grid_release(&grid);

        return error;
    }

    // set grid pointer
    *gridPointer = grid;

    return LINALGCL_SUCCESS;
}

// release solver grid
linalgcl_error_t ert_grid_release(ert_grid_t* gridPointer) {
    // check input
    if ((gridPointer == NULL) || (*gridPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get grid
    ert_grid_t grid = *gridPointer;

    // cleanup
    ert_mesh_release(&grid->mesh);
    linalgcl_sparse_matrix_release(&grid->system_matrix);
    linalgcl_matrix_release(&grid->gradient_matrix);
    linalgcl_sparse_matrix_release(&grid->gradient_matrix_sparse);
    linalgcl_matrix_release(&grid->sigma);
    linalgcl_matrix_release(&grid->area);
    ert_grid_program_release(&grid->program);

    // free struct
    free(grid);

    // set grid pointer to NULL
    *gridPointer = NULL;

    return LINALGCL_SUCCESS;
}

// init system matrix
linalgcl_error_t ert_grid_init_system_matrix(ert_grid_t grid,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue) {
    // check input
    if ((grid == NULL) || (matrix_program == NULL) ||
        (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // calc initial system matrix
    // create matrices
    linalgcl_matrix_t system_matrix, gradient_matrix, sigma_matrix;
    error = linalgcl_matrix_create(&system_matrix, context,
        grid->mesh->vertex_count, grid->mesh->vertex_count);
    error += linalgcl_matrix_create(&gradient_matrix, context,
        2 * grid->mesh->element_count, grid->mesh->vertex_count);
    error += linalgcl_matrix_unity(&sigma_matrix, context,
        2 * grid->mesh->element_count);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_grid_release(&grid);

        return LINALGCL_ERROR;
    }

    // calc gradient matrix
    linalgcl_matrix_data_t x[3], y[3];
    linalgcl_matrix_data_t id[3];
    ert_basis_t basis[3];

    // init matrices
    for (linalgcl_size_t i = 0; i < gradient_matrix->size_x; i++) {
        for (linalgcl_size_t j = 0; j < gradient_matrix->size_y; j++) {
            linalgcl_matrix_set_element(gradient_matrix, 0.0, i, j);
            linalgcl_matrix_set_element(grid->gradient_matrix, 0.0, j, i);
        }
    }

    linalgcl_matrix_data_t area;

    for (linalgcl_size_t k = 0; k < grid->mesh->element_count; k++) {
        // get vertices for element
        for (linalgcl_size_t i = 0; i < 3; i++) {
            linalgcl_matrix_get_element(grid->mesh->elements, &id[i], k, i);
            linalgcl_matrix_get_element(grid->mesh->vertices, &x[i], (linalgcl_size_t)id[i], 0);
            linalgcl_matrix_get_element(grid->mesh->vertices, &y[i], (linalgcl_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        ert_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        ert_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        ert_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        for (linalgcl_size_t i = 0; i < 3; i++) {
            linalgcl_matrix_set_element(gradient_matrix,
                basis[i]->gradient[0], 2 * k, (linalgcl_size_t)id[i]);
            linalgcl_matrix_set_element(gradient_matrix,
                basis[i]->gradient[1], 2 * k + 1, (linalgcl_size_t)id[i]);
            linalgcl_matrix_set_element(grid->gradient_matrix,
                basis[i]->gradient[0], (linalgcl_size_t)id[i], 2 * k);
            linalgcl_matrix_set_element(grid->gradient_matrix,
                basis[i]->gradient[1], (linalgcl_size_t)id[i], 2 * k + 1);
        }

        // calc area of element
        area = 0.5 * fabs((x[1] - x[0]) * (y[2] - y[0]) -
            (x[2] - x[0]) * (y[1] - y[0]));

        linalgcl_matrix_set_element(grid->area, area, k, 0);
        linalgcl_matrix_set_element(sigma_matrix, grid->sigma->host_data[k] * area,
            2 * k, 2 * k);
        linalgcl_matrix_set_element(sigma_matrix, grid->sigma->host_data[k] * area,
            2 * k + 1, 2 * k + 1);

        // cleanup
        ert_basis_release(&basis[0]);
        ert_basis_release(&basis[1]);
        ert_basis_release(&basis[2]);
    }

    // copy matrices to device
    error  = linalgcl_matrix_copy_to_device(gradient_matrix, queue, CL_TRUE);
    error += linalgcl_matrix_copy_to_device(grid->gradient_matrix, queue, CL_TRUE);
    error += linalgcl_matrix_copy_to_device(sigma_matrix, queue, CL_TRUE);
    error += linalgcl_matrix_copy_to_device(grid->sigma, queue, CL_TRUE);
    error += linalgcl_matrix_copy_to_device(grid->area, queue, CL_TRUE);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        linalgcl_matrix_release(&system_matrix);
        linalgcl_matrix_release(&gradient_matrix);
        linalgcl_matrix_release(&sigma_matrix);
        ert_grid_release(&grid);

        return LINALGCL_ERROR;
    }

    // calc system matrix
    linalgcl_matrix_t temp = NULL;
    error = linalgcl_matrix_create(&temp, context, grid->mesh->vertex_count,
        2 * grid->mesh->element_count);
    error += linalgcl_matrix_multiply(temp, grid->gradient_matrix, sigma_matrix,
        matrix_program, queue);
    error += linalgcl_matrix_multiply(system_matrix, temp, gradient_matrix,
        matrix_program, queue);
    clFinish(queue);

    // cleanup
    linalgcl_matrix_release(&sigma_matrix);
    linalgcl_matrix_release(&gradient_matrix);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        linalgcl_matrix_release(&system_matrix);
        ert_grid_release(&grid);

        return LINALGCL_ERROR;
    }

    // create sparse matrices
    error = linalgcl_sparse_matrix_create(&grid->system_matrix, system_matrix,
        matrix_program, context, queue);
    error += linalgcl_sparse_matrix_create(&grid->gradient_matrix_sparse,
        grid->gradient_matrix, matrix_program, context, queue);

    // cleanup
    linalgcl_matrix_release(&system_matrix);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_grid_release(&grid);

        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// update system matrix
linalgcl_error_t ert_grid_update_system_matrix(ert_grid_t grid,
    cl_command_queue queue) {
    // check input
    if ((grid == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(grid->program->kernel_update_system_matrix,
        0, sizeof(cl_mem), &grid->system_matrix->values->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        1, sizeof(cl_mem), &grid->system_matrix->column_ids->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        2, sizeof(cl_mem), &grid->gradient_matrix_sparse->values->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        3, sizeof(cl_mem), &grid->gradient_matrix_sparse->column_ids->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        4, sizeof(cl_mem), &grid->gradient_matrix->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        5, sizeof(cl_mem), &grid->sigma->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        6, sizeof(cl_mem), &grid->area->device_data);
    cl_error += clSetKernelArg(grid->program->kernel_update_system_matrix,
        7, sizeof(unsigned int), &grid->gradient_matrix->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global[2] = { grid->system_matrix->size_x, LINALGCL_BLOCK_SIZE };
    size_t local[2] = { LINALGCL_BLOCK_SIZE, LINALGCL_BLOCK_SIZE };

    cl_error = clEnqueueNDRangeKernel(queue,
        grid->program->kernel_update_system_matrix, 2,
        NULL, global, local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

linalgcl_matrix_data_t ert_grid_angle(linalgcl_matrix_data_t x, linalgcl_matrix_data_t y) {
    if (x > 0.0f) {
        return atan(y / x);
    }
    else if ((x < 0.0f) && (y >= 0.0f)) {
        return atan(y / x) + M_PI;
    }
    else if ((x < 0.0f) && (y < 0.0f)) {
        return atan(y / x) - M_PI;
    }
    else if ((x == 0.0f) && (y > 0.0f)) {
        return M_PI / 2.0f;
    }
    else if ((x == 0.0f) && (y < 0.0f)) {
        return - M_PI / 2.0f;
    }
    else {
        return 0.0f;
    }
}

linalgcl_matrix_data_t ert_grid_integrate_basis(linalgcl_matrix_data_t* node,
    linalgcl_matrix_data_t* left, linalgcl_matrix_data_t* right,
    linalgcl_matrix_data_t* start, linalgcl_matrix_data_t* end) {
    // integral
    linalgcl_matrix_data_t integral = 0.0f;

    // calc angle of node
    linalgcl_matrix_data_t angle = ert_grid_angle(node[0], node[1]);
    linalgcl_matrix_data_t radius = start[0] * start[0] + start[1] * start[1];

    // calc s parameter
    linalgcl_matrix_data_t sleft = radius * (ert_grid_angle(left[0], left[1]) - angle);
    linalgcl_matrix_data_t sright = radius * (ert_grid_angle(right[0], right[1]) - angle);
    linalgcl_matrix_data_t sstart = radius * (ert_grid_angle(start[0], start[1]) - angle);
    linalgcl_matrix_data_t send = radius * (ert_grid_angle(end[0], end[1]) - angle);

    // correct s parameter
    sleft -= (sleft > sright) && (sleft > 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sright += (sleft > sright) && (sleft < 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sstart -= (sstart > send) && (sstart > 0.0f) ? radius * 2.0f * M_PI : 0.0f;
    sstart += (sstart > send) && (sstart < 0.0f) ? radius * 2.0f * M_PI : 0.0f;

    // integrate left triangle
    if ((sstart < 0.0f) && (send > sleft)) {
        if ((send >= 0.0f) && (sstart <= sleft)) {
            integral = -0.5f * sleft;
        }
        else if ((send >= 0.0f) && (sstart > sleft)) {
            integral = -(sstart - 0.5 * sstart * sstart / sleft);
        }
        else if ((send < 0.0f) && (sstart <= sleft)) {
            integral = (send - 0.5 * send * send / sleft) -
                       (sleft - 0.5 * sleft * sleft / sleft);
        }
        else if ((send < 0.0f) && (sstart > sleft)) {
            integral = (send - 0.5 * send * send / sleft) -
                       (sstart - 0.5 * sstart * sstart / sleft);
        }
    }

    // integrate right triangle
    if ((send > 0.0f) && (sright > sstart)) {
        if ((sstart <= 0.0f) && (send >= sright)) {
            integral += 0.5f * sright;
        }
        else if ((sstart <= 0.0f) && (send < sright)) {
            integral += (send - 0.5f * send * send / sright);
        }
        else if ((sstart > 0.0f) && (send >= sright)) {
            integral += (sright - 0.5f * sright * sright / sright) -
                        (sstart - 0.5f * sstart * sstart / sright);
        }
        else if ((sstart > 0.0f) && (send < sright)) {
            integral += (send - 0.5f * send * send / sright) -
                        (sstart - 0.5f * sstart * sstart / sright);
        }
    }

    return integral;
}

// init exitation matrix
linalgcl_error_t ert_grid_init_exitation_matrix(ert_grid_t grid,
    ert_electrodes_t electrodes, cl_context context, cl_command_queue queue) {
    // check input
    if ((grid == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // create exitation_matrix
    error = linalgcl_matrix_create(&grid->exitation_matrix, context,
        grid->mesh->vertex_count, electrodes->count);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // calc electrode area
    linalgcl_matrix_data_t element_area = 2.0 * M_PI * grid->mesh->radius /
        (linalgcl_matrix_data_t)(electrodes->count * 2);

    // fill exitation_matrix matrix
    linalgcl_matrix_data_t id[3];
    linalgcl_matrix_data_t node[2], left[2], right[2];

    for (int i = 0; i < grid->mesh->boundary_count; i++) {
        for (int j = 0; j < electrodes->count; j++) {
            // get boundary node id
            linalgcl_matrix_get_element(grid->mesh->boundary, &id[0], i - 1 < 0 ? grid->mesh->boundary_count - 1 : i - 1, 0);
            linalgcl_matrix_get_element(grid->mesh->boundary, &id[1], i, 0);
            linalgcl_matrix_get_element(grid->mesh->boundary, &id[2], (i + 1) % grid->mesh->boundary_count, 0);

            // get coordinates
            linalgcl_matrix_get_element(grid->mesh->vertices, &left[0], (linalgcl_size_t)id[0], 0);
            linalgcl_matrix_get_element(grid->mesh->vertices, &left[1], (linalgcl_size_t)id[0], 1);
            linalgcl_matrix_get_element(grid->mesh->vertices, &node[0], (linalgcl_size_t)id[1], 0);
            linalgcl_matrix_get_element(grid->mesh->vertices, &node[1], (linalgcl_size_t)id[1], 1);
            linalgcl_matrix_get_element(grid->mesh->vertices, &right[0], (linalgcl_size_t)id[2], 0);
            linalgcl_matrix_get_element(grid->mesh->vertices, &right[1], (linalgcl_size_t)id[2], 1);

            // calc element
            linalgcl_matrix_set_element(grid->exitation_matrix,
                ert_grid_integrate_basis(node, left, right,
                    &electrodes->electrode_start[j * 2], &electrodes->electrode_end[j * 2]) / element_area,
                    (linalgcl_size_t)id[1], j);
        }
    }

    // upload matrix
    linalgcl_matrix_copy_to_device(grid->exitation_matrix, queue, CL_FALSE);

    return LINALGCL_SUCCESS;
}
