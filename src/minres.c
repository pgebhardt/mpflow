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
#include "minres.h"

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

// create new minres program
linalgcl_error_t ert_minres_solver_program_create(ert_minres_solver_program_t* programPointer,
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
    ert_minres_solver_program_t program = malloc(sizeof(ert_minres_solver_program_s));

    // check success
    if (program == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    program->program = NULL;

    // read program file
    // open file
    FILE* file = fopen(path, "r");

    // check success
    if (file == NULL) {
        // cleanup
        ert_minres_solver_program_release(&program);

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
        ert_minres_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // fread file
    if (fread(buffer, sizeof(char), length, file) != length) {
        // cleanup
        free(buffer);
        fclose(file);
        ert_minres_solver_program_release(&program);

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
        ert_minres_solver_program_release(&program);

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
        ert_minres_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // create kernel

    // set program pointer
    *programPointer = program;

    return LINALGCL_SUCCESS;
}

// release minres program
linalgcl_error_t ert_minres_solver_program_release(ert_minres_solver_program_t* programPointer) {
    // check input
    if ((programPointer == NULL) || (*programPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get program
    ert_minres_solver_program_t program = *programPointer;

    if (program->program != NULL) {
        clReleaseProgram(program->program);
    }

    // free struct
    free(program);

    // set program pointer to NULL
    *programPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create minres solver
linalgcl_error_t ert_minres_solver_create(ert_minres_solver_t* solverPointer,
    ert_grid_t grid, cl_context context, cl_device_id device_id, cl_command_queue queue) {
    // check input
    if ((solverPointer == NULL) || (grid == NULL) || (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create solver struct
    ert_minres_solver_t solver = malloc(sizeof(ert_minres_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    solver->grid = grid;
    solver->residuum = NULL;
    solver->projection[0] = NULL;
    solver->projection[1] = NULL;
    solver->projection[2] = NULL;
    solver->solution[0] = NULL;
    solver->solution[1] = NULL;
    solver->solution[2] = NULL;
    solver->program = NULL;

    // create matrices
    error  = linalgcl_matrix_create(&solver->residuum, context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->projection[0], context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->projection[1], context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->projection[2], context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->solution[0], context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->solution[1], context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->solution[2], context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->temp_matrix, context,
        solver->grid->mesh->vertex_count, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_minres_solver_release(&solver);

        return error;
    }

    // copy data to device
    error  = linalgcl_matrix_copy_to_device(solver->residuum, queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->projection[0], queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->projection[1], queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->projection[2], queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->solution[0], queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->solution[1], queue, CL_FALSE);
    error |= linalgcl_matrix_copy_to_device(solver->solution[2], queue, CL_TRUE);
    error |= linalgcl_matrix_copy_to_device(solver->temp_matrix, queue, CL_TRUE);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_minres_solver_release(&solver);

        return error;
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALGCL_SUCCESS;
}

// release solver
linalgcl_error_t ert_minres_solver_release(ert_minres_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get solver
    ert_minres_solver_t solver = *solverPointer;

    // release matrices
    linalgcl_matrix_release(&solver->residuum);
    linalgcl_matrix_release(&solver->projection[0]);
    linalgcl_matrix_release(&solver->projection[1]);
    linalgcl_matrix_release(&solver->projection[2]);
    linalgcl_matrix_release(&solver->solution[0]);
    linalgcl_matrix_release(&solver->solution[1]);
    linalgcl_matrix_release(&solver->solution[2]);
    linalgcl_matrix_release(&solver->temp_matrix);

    // release program
    ert_minres_solver_program_release(&solver->program);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// solve minres
linalgcl_error_t ert_minres_solver_solve(ert_minres_solver_t solver,
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
    // calc residuum r = b - A * x
    error  = linalgcl_sparse_matrix_vector_multiply(solver->residuum, solver->grid->system_matrix, x,
        matrix_program, queue);
    error |= linalgcl_matrix_scalar_multiply(solver->residuum, solver->residuum, -1.0,
        matrix_program, queue);
    error |= linalgcl_matrix_add(solver->residuum, solver->residuum, f,
        matrix_program, queue);
    clFinish(queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // p0 = r
    error  = linalgcl_matrix_copy(solver->projection[0], solver->residuum,
        queue, CL_FALSE);

    // s0 = A * p0 = A * r
    error |= linalgcl_sparse_matrix_vector_multiply(solver->solution[0],
        solver->grid->system_matrix, solver->residuum,
        matrix_program, queue);

    // p1 = p0 = r
    error |= linalgcl_matrix_copy(solver->projection[1], solver->residuum,
        queue, CL_TRUE);

    // s1 = s0
    error |= linalgcl_matrix_copy(solver->solution[1], solver->solution[0],
        queue, CL_TRUE);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // iterate
    linalgcl_matrix_data_t alpha, beta1, beta2, temp_number;
    for (linalgcl_size_t i = 0; i < 30; i++) {
        // update memory p2, p1 = p1, p0  s2, s1 = s1, s0
        linalgcl_matrix_copy(solver->projection[2], solver->projection[1], queue, CL_TRUE);
        linalgcl_matrix_copy(solver->projection[1], solver->projection[0], queue, CL_TRUE);
        linalgcl_matrix_copy(solver->solution[2], solver->solution[1], queue, CL_TRUE);
        linalgcl_matrix_copy(solver->solution[1], solver->solution[0], queue, CL_TRUE);

        // calc alpha on CPU
        clFinish(queue);
        linalgcl_matrix_copy_to_host(solver->residuum, queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(solver->solution[1], queue, CL_TRUE);

        alpha = 0.0;
        temp_number = 0.0;
        for (linalgcl_size_t k = 0; k < solver->residuum->size_x; k++) {
            alpha += solver->residuum->host_data[k] * solver->solution[1]->host_data[k];
            temp_number += solver->solution[1]->host_data[k] * solver->solution[1]->host_data[k];
        }
        alpha = alpha / temp_number;
        printf("iteration: %d, alpha: %f\n", i, alpha);

        // continue on GPU
        // update x
        linalgcl_matrix_scalar_multiply(solver->temp_matrix, solver->projection[1], alpha,
            matrix_program, queue);
        linalgcl_matrix_add(x, x, solver->temp_matrix, matrix_program, queue);

        // update residuum
        linalgcl_matrix_scalar_multiply(solver->temp_matrix, solver->solution[1], -alpha,
            matrix_program, queue);
        linalgcl_matrix_add(solver->residuum, solver->residuum, solver->temp_matrix,
            matrix_program, queue);

        // update projection
        linalgcl_matrix_copy(solver->projection[0], solver->solution[1], queue, CL_FALSE);
        linalgcl_sparse_matrix_vector_multiply(solver->solution[0], solver->grid->system_matrix,
            solver->solution[1], matrix_program, queue);

        // calc beta1 on CPU
        clFinish(queue);
        linalgcl_matrix_copy_to_host(solver->solution[0], queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(solver->solution[1], queue, CL_TRUE);

        beta1 = 0.0;
        temp_number = 0.0;
        for (linalgcl_size_t k = 0; k < solver->solution[0]->size_x; k++) {
            beta1 += solver->solution[0]->host_data[k] * solver->solution[1]->host_data[k];
            temp_number += solver->solution[1]->host_data[k] * solver->solution[1]->host_data[k];
        }
        beta1 = beta1 / temp_number;
        printf("iteration: %d, beta1: %f\n", i, beta1);

        // continue on GPU
        // update projection
        linalgcl_matrix_scalar_multiply(solver->temp_matrix, solver->projection[1], -beta1,
            matrix_program, queue);
        linalgcl_matrix_add(solver->projection[0], solver->projection[0], solver->temp_matrix,
            matrix_program, queue);

        // update solution
        linalgcl_matrix_scalar_multiply(solver->temp_matrix, solver->solution[1], -beta1,
            matrix_program, queue);
        linalgcl_matrix_add(solver->solution[0], solver->solution[0], solver->temp_matrix,
            matrix_program, queue);

        if (i > 0) {
            // calc beta2 on CPU
            clFinish(queue);
            linalgcl_matrix_copy_to_host(solver->solution[0], queue, CL_TRUE);
            linalgcl_matrix_copy_to_host(solver->solution[2], queue, CL_TRUE);

            beta2 = 0.0;
            temp_number = 0.0;
            for (linalgcl_size_t k = 0; k < solver->solution[0]->size_x; k++) {
                beta2 += solver->solution[0]->host_data[k] * solver->solution[2]->host_data[k];
                temp_number += solver->solution[2]->host_data[k] * solver->solution[2]->host_data[k];
            }
            beta2 = beta2 / temp_number;
            printf("iteration: %d, beta2: %f\n", i, beta2);

            // continue on GPU
            // update projection
            linalgcl_matrix_scalar_multiply(solver->temp_matrix, solver->projection[2], -beta2,
                matrix_program, queue);
            linalgcl_matrix_add(solver->projection[0], solver->projection[0], solver->temp_matrix,
                matrix_program, queue);

            // update solution
            linalgcl_matrix_scalar_multiply(solver->temp_matrix, solver->solution[2], -beta2,
                matrix_program, queue);
            linalgcl_matrix_add(solver->solution[0], solver->solution[0], solver->temp_matrix,
                matrix_program, queue);
        }

        // sync iteration
        clFinish(queue);
    }

    return LINALGCL_SUCCESS;

}
