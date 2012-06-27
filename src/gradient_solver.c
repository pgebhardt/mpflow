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
#include "gradient_solver.h"

// create new grid program
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
    program->kernel_unfold_system_matrix = NULL;
    program->kernel_regulize_system_matrix = NULL;

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
    linalgcl_size_t length = 0;
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
        (const char**)&buffer, NULL, &cl_error);
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
    program->kernel_unfold_system_matrix = clCreateKernel(program->program,
        "unfold_system_matrix", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_gradient_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    program->kernel_regulize_system_matrix = clCreateKernel(
        program->program, "regulize_system_matrix", &cl_error);

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

// release grid program
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

    if (program->kernel_unfold_system_matrix != NULL) {
        clReleaseKernel(program->kernel_unfold_system_matrix);
    }

    if (program->kernel_regulize_system_matrix != NULL) {
        clReleaseKernel(program->kernel_regulize_system_matrix);
    }

    // free struct
    free(program);

    // set program pointer to NULL
    *programPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create solver
linalgcl_error_t ert_gradient_solver_create(ert_gradient_solver_t* solverPointer,
    ert_grid_t grid, cl_context context, cl_device_id device_id, cl_command_queue queue) {
    // check input
    if ((solverPointer == NULL) || (grid == NULL) || (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create grid struct
    ert_gradient_solver_t solver = malloc(sizeof(ert_gradient_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    solver->grid = grid;
    solver->system_matrix = NULL;;
    solver->regulized_matrix;
    solver->residuum = NULL;
    solver->projection = NULL;
    solver->x = NULL;
    solver->temp[0] = NULL;
    solver->temp[1] = NULL;
    solver->program = NULL;

    // set solver pointer
    *solverPointer = solver;

    return LINALGCL_SUCCESS;
}

// release solver
linalgcl_error_t ert_gradient_solver_release(ert_gradient_solver_t* solverPointer) {

}

// unfold system matrix
linalgcl_error_t ert_gradient_solver_unfold_system_matrix(ert_gradient_solver_t solver,
    linalgcl_matrix_t result, cl_command_queue queue) {

}

// regulize system matrix
linalgcl_error_t ert_gradient_solver_regulize_system_matrix(ert_gradient_solver_t solver,
    linalgcl_matrix_data_t lambda, linalgcl_matrix_program_t matrix_program,
    cl_command_queue queue) {

}

// solve conjugate gradient
linalgcl_error_t ert_gradient_solver_solve(ert_gradient_solver_t solver,
    linalgcl_matrix_t initial_guess, linalgcl_matrix_t f,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (initial_guess == NULL) || (f == NULL) || 
        (matrix_program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    linalgcl_matrix_t r, p, temp1, temp2, R, A;
    linalgcl_matrix_data_t alpha, beta, temp3, rnorm;

    // create matrices
    error += linalgcl_matrix_create(&r, context, x->size_x, 1);
    error += linalgcl_matrix_create(&p, context, x->size_x, 1);
    error += linalgcl_matrix_create(&temp1, context, x->size_x, 1);
    error += linalgcl_matrix_create(&temp2, context, x->size_x, 1);
    error += linalgcl_matrix_create(&R, context,
        grid->system_matrix->size_x, grid->system_matrix->size_y);
    error += linalgcl_matrix_create(&A, context,
        grid->system_matrix->size_x, grid->system_matrix->size_y);

    // copy to device
    linalgcl_matrix_copy_to_device(r, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(p, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(temp1, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(temp2, queue, CL_TRUE);

    // regulize system matrix
    linalgcl_matrix_copy_to_device(A, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(R, queue, CL_TRUE);
    ert_grid_unfold_system_matrix(grid, A, grid_program, queue);
    clFinish(queue);
    linalgcl_matrix_multiply(matrix_program, queue, R, A, A);
    clFinish(queue);
    ert_grid_regulize_system_matrix(grid, R, 0.0001, grid_program,
        queue);
    clFinish(queue);

    linalgcl_matrix_copy_to_host(R, queue, CL_TRUE);
    linalgcl_matrix_save("A.txt", R);

    // regulize f
    linalgcl_matrix_copy_to_host(f, queue, CL_TRUE);
    linalgcl_sparse_matrix_vector_multiply(matrix_program, queue,
        temp1, grid->system_matrix, f);
    linalgcl_matrix_copy(matrix_program, queue, f, temp1);

    // set data
    // calc r0 = f - A * x0
    error += linalgcl_matrix_scalar_multiply(matrix_program, queue, temp1, x, -1.0);

    error += linalgcl_matrix_multiply(matrix_program, queue, temp2,
        R, temp1);

    error += linalgcl_matrix_add(matrix_program, queue, r, f, temp2);
    clFinish(queue);

    // init p0
    error += linalgcl_matrix_copy(matrix_program, queue, p, r);
    clFinish(queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // iteration
    for (linalgcl_size_t i = 0; i < 1000; i++) {
        // calc R * p
        linalgcl_matrix_multiply(matrix_program, queue, temp1, R, p);
        clFinish(queue);
        linalgcl_matrix_copy_to_host(temp1, queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(p, queue, CL_TRUE);

        // calc rnorm
        linalgcl_matrix_copy_to_host(r, queue, CL_TRUE);
        temp3 = 0.0;
        rnorm = 0.0;

        for (linalgcl_size_t k = 0; k < r->size_x; k++) {
            rnorm += r->host_data[k] * r->host_data[k];
            temp3 += p->host_data[k] * temp1->host_data[k];
        }
        alpha = rnorm / temp3;

        // check error
        if (sqrt(rnorm) <= 0.001) {
            break;
        }

        // correct x
        linalgcl_matrix_scalar_multiply(matrix_program, queue, temp2, p, alpha);
        linalgcl_matrix_add(matrix_program, queue, x, x, temp2);

        // correct r
        linalgcl_matrix_scalar_multiply(matrix_program, queue, temp2, temp1, -alpha);
        linalgcl_matrix_add(matrix_program, queue, r, r, temp2);
        clFinish(queue);

        linalgcl_matrix_copy_to_host(r, queue, CL_TRUE);

        // calc beta
        beta = 0.0;
        for (linalgcl_size_t k = 0; k < r->size_x; k++) {
            beta += r->host_data[k] * r->host_data[k];
        }
        beta = beta / rnorm;

        // calc new p
        linalgcl_matrix_scalar_multiply(matrix_program, queue, temp1, p, beta);
        linalgcl_matrix_add(matrix_program, queue, p, r, temp1);
        clFinish(queue);
    }

    // cleanup
    linalgcl_matrix_release(&r);
    linalgcl_matrix_release(&p);
    linalgcl_matrix_release(&temp1);
    linalgcl_matrix_release(&temp2);
    linalgcl_matrix_release(&R);
    linalgcl_matrix_release(&A);

    return LINALGCL_SUCCESS;

}
