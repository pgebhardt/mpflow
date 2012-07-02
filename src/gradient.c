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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <linalgcl/linalgcl.h>
#include "basis.h"
#include "mesh.h"
#include "grid.h"
#include "gradient.h"

static void print_matrix(linalgcl_matrix_t matrix) {
    if (matrix == NULL) {
        return;
    }

    // value memory
    linalgcl_matrix_data_t value = 0.0;

    for (linalgcl_size_t i = 0; i < matrix->size_x; i++) {
        for (linalgcl_size_t j = 0; j < matrix->size_y; j++) {
            // get value
            linalgcl_matrix_get_element(matrix, &value, i, j);

            printf("%f, ", value);
        }
        printf("\n");
    }
}

// create new gradient program
linalgcl_error_t ert_gradient_solver_program_create(ert_gradient_solver_program_t* programPointer,
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
    ert_gradient_solver_program_t program = malloc(sizeof(ert_gradient_solver_program_s));

    // check success
    if (program == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    program->program = NULL;
    program->kernel_regularize_system_matrix = NULL;
    program->kernel_update_vector = NULL;

    // read program file
    // open file
    FILE* file = fopen(path, "r");

    // check success
    if (file == NULL) {
        // cleanup
        ert_gradient_solver_program_release(&program);

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
        ert_gradient_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // fread file
    if (fread(buffer, sizeof(char), length, file) != length) {
        // cleanup
        free(buffer);
        fclose(file);
        ert_gradient_solver_program_release(&program);

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
        ert_gradient_solver_program_release(&program);

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
        ert_gradient_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // create kernel
    program->kernel_regularize_system_matrix = clCreateKernel(program->program,
        "regularize_system_matrix", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_gradient_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    program->kernel_update_vector = clCreateKernel(program->program,
        "update_vector", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_gradient_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // set program pointer
    *programPointer = program;

    return LINALGCL_SUCCESS;
}

// release gradient program
linalgcl_error_t ert_gradient_solver_program_release(ert_gradient_solver_program_t* programPointer) {
    // check input
    if ((programPointer == NULL) || (*programPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get program
    ert_gradient_solver_program_t program = *programPointer;

    if (program->program != NULL) {
        clReleaseProgram(program->program);
    }

    if (program->kernel_regularize_system_matrix != NULL) {
        clReleaseKernel(program->kernel_regularize_system_matrix);
    }

    if (program->kernel_update_vector != NULL) {
        clReleaseKernel(program->kernel_update_vector);
    }

    // free struct
    free(program);

    // set program pointer to NULL
    *programPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create gradient solver
linalgcl_error_t ert_gradient_solver_create(ert_gradient_solver_t* solverPointer,
    linalgcl_size_t size, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_device_id device_id, cl_command_queue queue) {
    // check input
    if ((solverPointer == NULL) || (size == 0) || (matrix_program == NULL) ||
        (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create solver struct
    ert_gradient_solver_t solver = malloc(sizeof(ert_gradient_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    solver->system_matrix = NULL;
    solver->residuum = NULL;
    solver->projection = NULL;
    solver->rsold = NULL;
    solver->rsnew = NULL;
    solver->temp_matrix = NULL;
    solver->temp_vector = NULL;
    solver->temp_number = NULL;
    solver->program = NULL;

    // create matrices
    error  = linalgcl_matrix_create(&solver->system_matrix, context, size, size);
    error |= linalgcl_matrix_create(&solver->residuum, context, size, 1);
    error |= linalgcl_matrix_create(&solver->projection, context, size, 1);
    error |= linalgcl_matrix_create(&solver->rsold, context, 1, 1);
    error |= linalgcl_matrix_create(&solver->rsnew, context, 1, 1);
    error |= linalgcl_matrix_create(&solver->temp_matrix, context, size, size);
    error |= linalgcl_matrix_create(&solver->temp_vector, context, size, 1);
    error |= linalgcl_matrix_create(&solver->temp_number, context, 1, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_gradient_solver_release(&solver);

        return error;
    }

    // copy data to device
    error  = linalgcl_matrix_copy_to_device(solver->system_matrix, queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->residuum, queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->projection, queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->temp_matrix, queue, CL_TRUE);
    error |= linalgcl_matrix_copy_to_device(solver->temp_vector, queue, CL_TRUE);
    error |= linalgcl_matrix_copy_to_device(solver->temp_number, queue, CL_TRUE);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_gradient_solver_release(&solver);

        return error;
    }

    // create program
    error = ert_gradient_solver_program_create(&solver->program, context,
        device_id, "src/gradient.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_gradient_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCL_SUCCESS;
}

// release solver
linalgcl_error_t ert_gradient_solver_release(ert_gradient_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get solver
    ert_gradient_solver_t solver = *solverPointer;

    // release matrices
    linalgcl_matrix_release(&solver->system_matrix);
    linalgcl_matrix_release(&solver->residuum);
    linalgcl_matrix_release(&solver->projection);
    linalgcl_matrix_release(&solver->rsold);
    linalgcl_matrix_release(&solver->rsnew);
    linalgcl_matrix_release(&solver->temp_matrix);
    linalgcl_matrix_release(&solver->temp_vector);
    linalgcl_matrix_release(&solver->temp_number);

    // release program
    ert_gradient_solver_program_release(&solver->program);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// regularize_system_matrix
linalgcl_error_t ert_gradient_solver_regularize_system_matrix(ert_gradient_solver_t solver,
    linalgcl_matrix_data_t sigma, linalgcl_matrix_program_t matrix_program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (matrix_program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(solver->program->kernel_regularize_system_matrix,
        0, sizeof(cl_mem), &solver->temp_matrix->device_data);
    cl_error |= clSetKernelArg(solver->program->kernel_regularize_system_matrix,
        1, sizeof(cl_mem), &solver->system_matrix->device_data);
    cl_error |= clSetKernelArg(solver->program->kernel_regularize_system_matrix,
        2, sizeof(linalgcl_size_t), &solver->system_matrix->size_y);
    cl_error |= clSetKernelArg(solver->program->kernel_regularize_system_matrix,
        3, sizeof(linalgcl_matrix_data_t), &sigma);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global[2] = { solver->system_matrix->size_x, solver->system_matrix->size_y };
    size_t local[2] = { LINALGCL_BLOCK_SIZE, LINALGCL_BLOCK_SIZE };

    cl_error = clEnqueueNDRangeKernel(queue, solver->program->kernel_regularize_system_matrix,
        2, NULL, global, local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // copy matrices
    linalgcl_matrix_copy(solver->system_matrix, solver->temp_matrix, queue, CL_TRUE);

    return LINALGCL_SUCCESS;
}

// update vector
linalgcl_error_t ert_gradient_update_vector(ert_gradient_solver_t solver,
    linalgcl_matrix_t result, linalgcl_matrix_t x1, linalgcl_matrix_data_t sign,
    linalgcl_matrix_t x2, linalgcl_matrix_t r1, linalgcl_matrix_t r2, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (result == NULL) || (x1 == NULL) || (x2 == NULL) ||
        (r1 == NULL) || (r2 == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(solver->program->kernel_update_vector,
        0, sizeof(cl_mem), &result->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_vector,
        1, sizeof(cl_mem), &x1->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_vector,
        2, sizeof(linalgcl_matrix_data_t), &sign);
    cl_error += clSetKernelArg(solver->program->kernel_update_vector,
        3, sizeof(cl_mem), &x2->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_vector,
        4, sizeof(cl_mem), &r1->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_vector,
        5, sizeof(cl_mem), &r2->device_data);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = result->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, solver->program->kernel_update_vector,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// solve gradient
linalgcl_error_t ert_gradient_solver_solve(ert_gradient_solver_t solver,
    linalgcl_matrix_t x, linalgcl_matrix_t f,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (x == NULL) || (f == NULL) || 
        (matrix_program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init matrices
    // calc residuum r = f - A * x
    error  = linalgcl_matrix_multiply(solver->residuum, solver->system_matrix, x,
        matrix_program, queue);
    error |= linalgcl_matrix_scalar_multiply(solver->residuum, solver->residuum, -1.0,
        matrix_program, queue);
    error |= linalgcl_matrix_add(solver->residuum, solver->residuum, solver->temp_vector,
        matrix_program, queue);
    clFinish(queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // p = r
    error  = linalgcl_matrix_copy(solver->projection, solver->residuum,
        queue, CL_FALSE);

    // calc rsold
    error += linalgcl_matrix_vector_dot_product(solver->rsold, solver->residuum,
        solver->residuum, matrix_program, queue);
    linalgcl_matrix_copy_to_host(solver->rsold, queue, CL_TRUE);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // iterate
    linalgcl_matrix_data_t alpha, beta, rsold;
    for (linalgcl_size_t i = 0; i < 1000; i++) {
        // check error
        linalgcl_matrix_copy_to_host(solver->rsold, queue, CL_FALSE);

        if (sqrt(solver->rsold->host_data[0]) / solver->system_matrix->size_x <= 1e-4) {
            printf("stopped after %d iterations!\n", i);
            break;
        }

        // calc A * p
        linalgcl_matrix_multiply(solver->temp_vector, solver->system_matrix,
            solver->projection, matrix_program, queue);

        // calc p * A * p
        linalgcl_matrix_vector_dot_product(solver->temp_number, solver->projection,
            solver->temp_vector, matrix_program, queue);

        // update residuum
        ert_gradient_update_vector(solver, solver->residuum, solver->residuum, -1.0,
            solver->temp_vector, solver->rsold, solver->temp_number, queue);

        // update x
        ert_gradient_update_vector(solver, x, x, 1.0, solver->projection, solver->rsold,
            solver->temp_number, queue);

        // calc rsnew
        linalgcl_matrix_vector_dot_product(solver->rsnew, solver->residuum,
            solver->residuum, matrix_program, queue);

        // update projection
        ert_gradient_update_vector(solver, solver->projection, solver->residuum, 1.0,
            solver->projection, solver->rsnew, solver->rsold, queue);

        // update rsold
        linalgcl_matrix_copy(solver->rsold, solver->rsnew, queue, CL_TRUE);
    }

    return LINALGCL_SUCCESS;

}
