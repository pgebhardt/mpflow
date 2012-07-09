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
#include "electrodes.h"
#include "grid.h"
#include "gradient.h"
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
    program->kernel_copy_to_column = NULL;
    program->kernel_copy_from_column = NULL;

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
        (const char**)&buffer, &length, &cl_error);
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
    program->kernel_copy_to_column = clCreateKernel(program->program,
        "copy_to_column", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    program->kernel_copy_from_column = clCreateKernel(program->program,
        "copy_from_column", &cl_error);

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

    if (program->kernel_copy_to_column != NULL) {
        clReleaseKernel(program->kernel_copy_to_column);
    }

    if (program->kernel_copy_from_column != NULL) {
        clReleaseKernel(program->kernel_copy_from_column);
    }

    // free struct
    free(program);

    // set program pointer to NULL
    *programPointer = NULL;

    return LINALGCL_SUCCESS;
}

// create solver
linalgcl_error_t ert_solver_create(ert_solver_t* solverPointer,
    ert_mesh_t mesh, ert_electrodes_t electrodes,
    linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_device_id device_id, cl_command_queue queue) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (matrix_program == NULL) || (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init solver pointer
    *solverPointer = NULL;

    // create struct
    ert_solver_t solver = malloc(sizeof(ert_solver_s));

    // check success
    if (solver == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    solver->program = NULL;
    solver->grid = NULL;
    solver->gradient_solver = NULL;
    solver->electrodes = electrodes;
    solver->voltage_calculation = NULL;
    solver->sigma = NULL;
    solver->current = NULL;
    solver->voltage = NULL;
    solver->f = NULL;

    // create program
    error = ert_solver_program_create(&solver->program, context, device_id, "src/solver.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // create grid
    error  = ert_grid_create(&solver->grid, matrix_program, mesh, context,
        device_id, queue);
    error |= ert_grid_init_exitation_matrix(solver->grid, solver->electrodes, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // create gradient solver
    error = ert_gradient_solver_create(&solver->gradient_solver,
        solver->grid->system_matrix, solver->grid->mesh->vertex_count,
        matrix_program, context, device_id, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // set sigma matrix
    solver->sigma = solver->grid->sigma;

    // create matrices
    error  = linalgcl_matrix_create(&solver->voltage_calculation, context,
        solver->grid->exitation_matrix->size_y, solver->grid->exitation_matrix->size_x);
    error |= linalgcl_matrix_create(&solver->current, context,
        solver->electrodes->count, 1);
    error |= linalgcl_matrix_create(&solver->voltage, context,
        solver->electrodes->count, 1);
    error |= linalgcl_matrix_create(&solver->f, context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->phi, context,
        solver->grid->mesh->vertex_count, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(solver->voltage_calculation, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->current, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->voltage, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->f, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->phi, queue, CL_TRUE);

    // calc voltage calculation matrix
    linalgcl_matrix_transpose(solver->voltage_calculation, solver->grid->exitation_matrix,  
        matrix_program, queue);

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
    ert_solver_program_release(&solver->program);
    ert_grid_release(&solver->grid);
    ert_gradient_solver_release(&solver->gradient_solver);
    ert_electrodes_release(&solver->electrodes);
    linalgcl_matrix_release(&solver->voltage_calculation);
    linalgcl_matrix_release(&solver->current);
    linalgcl_matrix_release(&solver->voltage);
    linalgcl_matrix_release(&solver->f);
    linalgcl_matrix_release(&solver->phi);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// forward solving
linalgcl_error_t ert_solver_forward_solve(ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (matrix_program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // calc right side
    error = linalgcl_matrix_multiply(solver->f, solver->grid->exitation_matrix,
    solver->current, matrix_program, queue);

    // solve for phi
    error |= ert_gradient_solver_solve_singular(solver->gradient_solver,
        solver->phi, solver->f, 1E-5, matrix_program, queue);

    // calc voltage
    error |= linalgcl_matrix_multiply(solver->voltage, solver->voltage_calculation,
        solver->phi, matrix_program, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    return LINALGCL_SUCCESS;
}
