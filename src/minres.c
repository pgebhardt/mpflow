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

    // set solver pointer
    *solverPointer = solver;

    return LINALGCL_SUCCESS;
}

// release solver
linalgcl_error_t ert_minres_solver_release(ert_minres_solver_t* solverPointer) {

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
    for (linalgcl_size_t i = 0; i < 1; i++) {

    }

    return LINALGCL_SUCCESS;

}
