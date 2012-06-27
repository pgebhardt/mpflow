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

// create gradient solver
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
    solver->regulized_matrix = NULL;
    solver->residuum = NULL;
    solver->projection = NULL;
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

// regulize system matrix
linalgcl_error_t ert_gradient_solver_regulize_system_matrix(ert_gradient_solver_t solver,
    linalgcl_matrix_data_t lambda, linalgcl_matrix_program_t matrix_program,
    cl_command_queue queue) {

}

// solve conjugate gradient
linalgcl_error_t ert_gradient_solver_solve(ert_gradient_solver_t solver,
    linalgcl_matrix_t x, linalgcl_matrix_t f,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (x == NULL) || (f == NULL) || 
        (matrix_program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // memory
    linalgcl_matrix_data_t alpha, beta, temp3, rnorm;

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // regulize system matrix
    ert_gradient_solver_regulize_system_matrix(solver, 0.0001, matrix_program, queue);

    // regulize f
    linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, solver->temp[0], solver->grid->system_matrix, f);
    linalgcl_matrix_copy(matrix_program, queue, f, solver->temp[0]);

    // calc r0 = f - A * x0
    error += linalgcl_matrix_scalar_multiply(matrix_program, queue, solver->temp[0], x, -1.0);
    error += linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, solver->temp[1], solver->regulized_matrix, solver->temp[0]);
    error += linalgcl_matrix_add(matrix_program, queue, solver->residuum, f, solver->temp[1]);

    // init p0
    error += linalgcl_matrix_copy(matrix_program, queue, solver->projection, solver->residuum);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // iteration
    for (linalgcl_size_t i = 0; i < 1000; i++) {
        // calc R * p
        linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, solver->temp[0],
            solver->regulized_matrix, solver->projection);

        // copy data to host and continue on CPU
        clFinish(queue);
        linalgcl_matrix_copy_to_host(solver->temp[0], queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(solver->projection, queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(solver->residuum, queue, CL_TRUE);

        // calc rnorm
        temp3 = 0.0;
        rnorm = 0.0;

        for (linalgcl_size_t k = 0; k < solver->residuum->size_x; k++) {
            rnorm += solver->residuum->host_data[k] * solver->residuum->host_data[k];
            temp3 += solver->projection->host_data[k] * solver->temp[0]->host_data[k];
        }
        alpha = rnorm / temp3;

        // check error
        if (sqrt(rnorm) <= 0.001) {
            break;
        }

        // correct x
        linalgcl_matrix_scalar_multiply(matrix_program, queue, solver->temp[1],
            solver->projection, alpha);
        linalgcl_matrix_add(matrix_program, queue, x, x, solver->temp[1]);

        // correct r
        linalgcl_matrix_scalar_multiply(matrix_program, queue, solver->temp[1], solver->temp[0], -alpha);
        linalgcl_matrix_add(matrix_program, queue, solver->residuum, solver->residuum, solver->temp[1]);

        // copy data to host and continue on CPU
        clFinish(queue);
        linalgcl_matrix_copy_to_host(solver->residuum, queue, CL_TRUE);

        // calc beta
        beta = 0.0;
        for (linalgcl_size_t k = 0; k < solver->residuum->size_x; k++) {
            beta += solver->residuum->host_data[k] * solver->residuum->host_data[k];
        }
        beta = beta / rnorm;

        // calc new p
        linalgcl_matrix_scalar_multiply(matrix_program, queue, solver->temp[0],
            solver->projection, beta);
        linalgcl_matrix_add(matrix_program, queue, solver->projection,
            solver->residuum, solver->temp[0]);

        // finish step
        clFinish(queue);
    }

    return LINALGCL_SUCCESS;

}
