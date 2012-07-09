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
    ert_mesh_t mesh, ert_electrodes_t electrodes, linalgcl_size_t measurment_count,
    linalgcl_size_t drive_count, linalgcl_matrix_program_t matrix_program,
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
    solver->measurment_count = measurment_count;
    solver->drive_count = drive_count;
    solver->voltage_calculation = NULL;
    solver->sigma = NULL;
    solver->phi = NULL;
    solver->applied_phi = NULL;
    solver->lead_phi = NULL;
    solver->applied_f = NULL;
    solver->lead_f = NULL;

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
    error |= linalgcl_matrix_create(&solver->phi, context,
        solver->grid->mesh->vertex_count, 1);
    error |= linalgcl_matrix_create(&solver->applied_phi, context,
        solver->grid->mesh->vertex_count, solver->drive_count);
    error |= linalgcl_matrix_create(&solver->lead_phi, context,
        solver->grid->mesh->vertex_count, solver->measurment_count);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(solver->voltage_calculation, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->phi, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(solver->applied_phi, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(solver->lead_phi, queue, CL_TRUE);

    // create f matrix storage
    solver->applied_f = malloc(sizeof(linalgcl_matrix_t) * solver->drive_count);
    solver->lead_f = malloc(sizeof(linalgcl_matrix_t) * solver->measurment_count);

    // check success
    if ((solver->applied_f == NULL) || (solver->lead_f == NULL)) {
        // cleanup
        ert_solver_release(&solver);

        return LINALGCL_ERROR;
    }

    // calc voltage calculation matrix
    linalgcl_matrix_transpose(solver->voltage_calculation, solver->grid->exitation_matrix,  
        matrix_program, queue);

    // load measurment pattern and drive pattern
    linalgcl_matrix_t measurment_pattern, drive_pattern;
    error  = linalgcl_matrix_load(&measurment_pattern, context, "measurment_pattern.txt");
    error |= linalgcl_matrix_load(&drive_pattern, context, "drive_pattern.txt");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // copy to device
    linalgcl_matrix_copy_to_device(measurment_pattern, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(drive_pattern, queue, CL_TRUE);

    // calc excitaion matrices
    error = ert_solver_calc_excitaion(solver, drive_pattern, measurment_pattern,
        matrix_program, context, queue);

    // cleanup
    linalgcl_matrix_release(&measurment_pattern);
    linalgcl_matrix_release(&drive_pattern);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

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
    linalgcl_matrix_release(&solver->phi);
    linalgcl_matrix_release(&solver->applied_phi);
    linalgcl_matrix_release(&solver->lead_phi);

    /*if (solver->lead_f != NULL) {
        for (linalgcl_size_t i = 0; i < solver->measurment_count; i++) {
            linalgcl_matrix_release(&solver->lead_f[i]);
        }
        free(solver->lead_f);
    }*/

    if (solver->applied_f != NULL) {
        for (linalgcl_size_t i = 0; i < solver->drive_count; i++) {
            linalgcl_matrix_release(&solver->applied_f[i]);
        }
        free(solver->applied_f);
    }

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// copy to column
linalgcl_error_t ert_solver_copy_to_column(ert_solver_t solver,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (matrix == NULL) || (vector == NULL) ||
        (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(solver->program->kernel_copy_to_column,
        0, sizeof(cl_mem), &matrix->device_data);
    cl_error |= clSetKernelArg(solver->program->kernel_copy_to_column,
        1, sizeof(cl_mem), &vector->device_data);
    cl_error |= clSetKernelArg(solver->program->kernel_copy_to_column,
        2, sizeof(linalgcl_size_t), &column);
    cl_error |= clSetKernelArg(solver->program->kernel_copy_to_column,
        3, sizeof(linalgcl_size_t), &matrix->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = vector->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, solver->program->kernel_copy_to_column,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// copy from column
linalgcl_error_t ert_solver_copy_from_column(ert_solver_t solver,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (matrix == NULL) || (vector == NULL) ||
        (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(solver->program->kernel_copy_from_column,
        0, sizeof(cl_mem), &matrix->device_data);
    cl_error |= clSetKernelArg(solver->program->kernel_copy_from_column,
        1, sizeof(cl_mem), &vector->device_data);
    cl_error |= clSetKernelArg(solver->program->kernel_copy_from_column,
        2, sizeof(linalgcl_size_t), &column);
    cl_error |= clSetKernelArg(solver->program->kernel_copy_from_column,
        3, sizeof(linalgcl_size_t), &matrix->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = vector->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, solver->program->kernel_copy_from_column,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// calc excitaion
linalgcl_error_t ert_solver_calc_excitaion(ert_solver_t solver,
    linalgcl_matrix_t drive_pattern, linalgcl_matrix_t measurment_pattern,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (drive_pattern == NULL) || (measurment_pattern == NULL) ||
        (matrix_program == NULL) || (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // create current matrix
    linalgcl_matrix_t current;
    error = linalgcl_matrix_create(&current, context, solver->electrodes->count, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // create drive pattern
    for (linalgcl_size_t i = 0; i < solver->drive_count; i++) {
        // create matrix
        linalgcl_matrix_create(&solver->applied_f[i], context, solver->grid->mesh->vertex_count, 1);

        // get current pattern
        ert_solver_copy_from_column(solver, drive_pattern, current, i, queue);

        // calc f
        linalgcl_matrix_multiply(solver->applied_f[i], solver->grid->exitation_matrix,
            current, matrix_program, queue);
    }

    // cleanup
    linalgcl_matrix_release(&current);

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

    // solve drive patterns
    for (linalgcl_size_t i = 0; i < solver->drive_count; i++) {
        // copy current applied phi to vector
        error = ert_solver_copy_from_column(solver, solver->applied_phi,
            solver->phi, i, queue);

        // solve for phi
        error |= ert_gradient_solver_solve_singular(solver->gradient_solver,
            solver->phi, solver->applied_f[i], 1E-5, matrix_program, queue);

        // copy vector to applied phi
        error |= ert_solver_copy_to_column(solver, solver->applied_phi,
            solver->phi, i, queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            return error;
        }
    }

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    return LINALGCL_SUCCESS;
}
