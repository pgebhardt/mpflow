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
#include <sys/time.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <linalgcl/linalgcl.h>
#include "basis.h"
#include "mesh.h"
#include "solver.h"

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

// create solver
linalgcl_error_t ert_solver_create(ert_solver_t* solverPointer, ert_mesh_t mesh,
    cl_context context, cl_command_queue queue, cl_device_id device_id,
    linalgcl_matrix_program_t program) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (context == NULL) ||
        (queue == NULL) || (program == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init solver pointer to NULL
    *solverPointer = NULL;

    // create solver struct
    ert_solver_t solver = NULL;
    solver = malloc(sizeof(ert_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    solver->mesh = mesh;
    solver->system_matrix = NULL;
    solver->gradient_matrix = NULL;
    solver->gradient_matrix_transposed = NULL;
    solver->sigma_matrix = NULL;

    // create matrices
    error  = linalgcl_matrix_create(&solver->system_matrix, context,
        solver->mesh->vertex_count, solver->mesh->vertex_count);
    error += linalgcl_matrix_create(&solver->gradient_matrix, context,
        2 * solver->mesh->element_count, solver->mesh->vertex_count);
    error += linalgcl_matrix_create(&solver->gradient_matrix_transposed, context,
        solver->mesh->vertex_count, 2 * solver->mesh->element_count);
    error += linalgcl_matrix_unity(&solver->sigma_matrix, context,
        2 * solver->mesh->element_count);
    error += linalgcl_matrix_copy_to_device(solver->sigma_matrix, queue, CL_TRUE);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // calc gradient matrix
    linalgcl_matrix_data_t x[3], y[3];
    linalgcl_matrix_data_t id[3];
    ert_basis_t basis[3];

    // init matrices
    for (linalgcl_size_t i = 0; i < solver->gradient_matrix->size_x; i++) {
        for (linalgcl_size_t j = 0; j < solver->gradient_matrix->size_y; j++) {
            linalgcl_matrix_set_element(solver->gradient_matrix, 0.0, i, j);
            linalgcl_matrix_set_element(solver->gradient_matrix_transposed, 0.0, j, i);
        }
    }

    for (linalgcl_size_t k = 0; k < solver->mesh->element_count; k++) {
        // get vertices for element
        for (linalgcl_size_t i = 0; i < 3; i++) {
            linalgcl_matrix_get_element(solver->mesh->elements, &id[i], k, i);
            linalgcl_matrix_get_element(solver->mesh->vertices, &x[i], (linalgcl_size_t)id[i], 0);
            linalgcl_matrix_get_element(solver->mesh->vertices, &y[i], (linalgcl_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        ert_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        ert_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        ert_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        for (linalgcl_size_t i = 0; i < 3; i++) {
            linalgcl_matrix_set_element(solver->gradient_matrix,
                basis[i]->gradient[0], 2 * k, (linalgcl_size_t)id[i]);
            linalgcl_matrix_set_element(solver->gradient_matrix,
                basis[i]->gradient[1], 2 * k + 1, (linalgcl_size_t)id[i]);
            linalgcl_matrix_set_element(solver->gradient_matrix_transposed,
                basis[i]->gradient[0], (linalgcl_size_t)id[i], 2 * k);
            linalgcl_matrix_set_element(solver->gradient_matrix_transposed,
                basis[i]->gradient[1], (linalgcl_size_t)id[i], 2 * k + 1);
        }

        // cleanup
        ert_basis_release(&basis[0]);
        ert_basis_release(&basis[1]);
        ert_basis_release(&basis[2]);
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(solver->gradient_matrix, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(solver->gradient_matrix_transposed, queue, CL_TRUE);

    // calc sigma matrix
    // TODO: non uniform sigma

    // get start time
    struct timeval tv;
    double start;
    gettimeofday(&tv, NULL);

    // convert time
    start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // calc system matrix
    linalgcl_matrix_t temp = NULL;
    linalgcl_matrix_create(&temp, context, solver->mesh->vertex_count,
        2 * solver->mesh->element_count);
    error = linalgcl_matrix_multiply(program, queue, temp, solver->gradient_matrix_transposed,
        solver->sigma_matrix);
    error += linalgcl_matrix_multiply(program, queue, solver->system_matrix,
        temp, solver->gradient_matrix);

    clFinish(queue);

    // get end time
    gettimeofday(&tv, NULL);

    // convert time
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // print time
    printf("Full matrix time: %f s\n", end - start);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return LINALGCL_ERROR;
    }

    // Test of optimized system matrix assembly
    // create solver program
    ert_solver_program_t solver_program = NULL;
    error = ert_solver_program_create(&solver_program, context, device_id,
        "src/solver.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return LINALGCL_ERROR;
    }

    // create sigma vector
    linalgcl_matrix_t sigma = NULL;
    linalgcl_matrix_create(&sigma, context, solver->mesh->element_count, 1);

    // set uniform sigma
    for (linalgcl_size_t i = 0; i < sigma->size_x; i++) {
        linalgcl_matrix_set_element(sigma, 1.0, i, 0);
    }

    // copy to device
    linalgcl_matrix_copy_to_device(sigma, queue, CL_TRUE);

    // get start time
    gettimeofday(&tv, NULL);

    // convert time
    start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // update system matrix
    ert_solver_update_system_matrix(solver, sigma, solver_program, queue);

    // convert time
    end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // print time
    printf("Optimized time: %f s\n", end - start);

    // set solver pointer
    *solverPointer = solver;

    return LINALGCL_SUCCESS;
}

// release solver
linalgcl_error_t ert_solver_release(ert_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get solver
    ert_solver_t solver = *solverPointer;

    // cleanup
    ert_mesh_release(&solver->mesh);
    linalgcl_matrix_release(&solver->system_matrix);
    linalgcl_matrix_release(&solver->gradient_matrix);
    linalgcl_matrix_release(&solver->gradient_matrix_transposed);
    linalgcl_matrix_release(&solver->sigma_matrix);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create new solver program
linalgcl_error_t ert_solver_program_create(ert_solver_program_t* programPointer,
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
    ert_solver_program_t program = malloc(sizeof(ert_solver_program_s));

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
        ert_solver_program_release(&program);

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
        ert_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // fread file
    if (fread(buffer, sizeof(char), length, file) != length) {
        // cleanup
        free(buffer);
        fclose(file);
        ert_solver_program_release(&program);

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
        ert_solver_program_release(&program);

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
        ert_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // create kernel
    program->kernel_update_system_matrix = clCreateKernel(program->program,
        "update_system_matrix", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    // set program pointer
    *programPointer = program;

    return LINALGCL_SUCCESS;
}

// release solver program
linalgcl_error_t ert_solver_program_release(ert_solver_program_t* programPointer) {
    // check input
    if ((programPointer == NULL) || (*programPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get program
    ert_solver_program_t program = *programPointer;

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

// update system matrix
linalgcl_error_t ert_solver_update_system_matrix(ert_solver_t solver,
    linalgcl_matrix_t sigma, ert_solver_program_t program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (sigma == NULL) || (program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}
