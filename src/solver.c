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
#include <actor/actor.h>
#include "basis.h"
#include "mesh.h"
#include "electrodes.h"
#include "grid.h"
#include "conjugate.h"
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
    program->kernel_calc_jacobian = NULL;
    program->kernel_calc_gradient = NULL;

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

    program->kernel_calc_jacobian = clCreateKernel(program->program,
        "calc_jacobian", &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        // cleanup
        ert_solver_program_release(&program);

        return LINALGCL_ERROR;
    }

    program->kernel_calc_gradient = clCreateKernel(program->program,
        "calc_gradient", &cl_error);

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

    if (program->kernel_calc_jacobian != NULL) {
        clReleaseKernel(program->kernel_calc_jacobian);
    }

    if (program->kernel_calc_gradient != NULL) {
        clReleaseKernel(program->kernel_calc_gradient);
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
    cl_context context, cl_device_id device, cl_command_queue queue) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL) || (electrodes == NULL) ||
        (matrix_program == NULL) || (context == NULL) || (device == NULL) ||
        (queue == NULL)) {
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
    solver->programs = NULL;
    solver->grids = NULL;
    solver->conjugate_solver = NULL;
    solver->electrodes = electrodes;
    solver->measurment_count = measurment_count;
    solver->drive_count = drive_count;
    solver->voltage_calculation = NULL;
    solver->applied_phi = NULL;
    solver->lead_phi = NULL;
    solver->applied_f = NULL;
    solver->lead_f = NULL;
    solver->calculated_voltage = NULL;
    solver->measured_voltage = NULL;

    // create program
    solver->programs = malloc(sizeof(ert_solver_program_s) * 2);

    error  = ert_solver_program_create(&solver->programs[0], context, device,
        "src/solver.cl");
    error |= ert_solver_program_create(&solver->programs[1], context, device,
        "src/solver.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // create grids
    solver->grids = malloc(sizeof(ert_grid_s) * 2);

    error  = ert_grid_create(&solver->grids[0], matrix_program, mesh, context,
        device, queue);
    error |= ert_grid_create(&solver->grids[1], matrix_program, mesh, context,
        device, queue);

    error |= ert_grid_init_exitation_matrix(solver->grids[0], solver->electrodes, context,
        queue);
    error |= ert_grid_init_exitation_matrix(solver->grids[1], solver->electrodes, context,
        queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // create matrices
    error  = linalgcl_matrix_create(&solver->voltage_calculation, context,
        solver->measurment_count, solver->grids[0]->exitation_matrix->size_x);
    error |= linalgcl_matrix_create(&solver->applied_phi, context,
        mesh->vertex_count, solver->drive_count);
    error |= linalgcl_matrix_create(&solver->lead_phi, context,
        mesh->vertex_count, solver->measurment_count);
    error |= linalgcl_matrix_create(&solver->calculated_voltage, context,
        solver->measurment_count, solver->drive_count);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(solver->voltage_calculation, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->applied_phi, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->lead_phi, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->calculated_voltage, queue, CL_TRUE);

    // create f matrix storage
    solver->applied_f = malloc(sizeof(linalgcl_matrix_t) * solver->drive_count);
    solver->lead_f = malloc(sizeof(linalgcl_matrix_t) * solver->measurment_count);

    // check success
    if ((solver->applied_f == NULL) || (solver->lead_f == NULL)) {
        // cleanup
        ert_solver_release(&solver);

        return LINALGCL_ERROR;
    }

    // create conjugate solver
    solver->conjugate_solver = malloc(sizeof(ert_conjugate_solver_s) * 2);

    error  = ert_conjugate_solver_create(&solver->conjugate_solver[0],
        solver->grids[0]->system_matrix, mesh->vertex_count,
        matrix_program, context, device, queue);
    error |= ert_conjugate_solver_create(&solver->conjugate_solver[1],
        solver->grids[1]->system_matrix, mesh->vertex_count,
        matrix_program, context, device, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // load measurment pattern and drive pattern
    linalgcl_matrix_t measurment_pattern, drive_pattern;
    error  = linalgcl_matrix_load(&measurment_pattern, context, "input/measurment_pattern.txt");
    error |= linalgcl_matrix_load(&drive_pattern, context, "input/drive_pattern.txt");
    error |= linalgcl_matrix_load(&solver->measured_voltage, context, "input/measured_voltage.txt");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // copy to device
    linalgcl_matrix_copy_to_device(measurment_pattern, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(drive_pattern, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(solver->measured_voltage, queue, CL_TRUE);

    // calc excitaion matrices
    error = ert_solver_calc_excitaion(solver, drive_pattern, measurment_pattern,
        matrix_program, context, queue);

    // calc voltage calculation matrix
    linalgcl_matrix_t temp1, temp2;
    error |= linalgcl_matrix_create(&temp1, context,
        measurment_pattern->size_y, measurment_pattern->size_x);
    error  = linalgcl_matrix_create(&temp2, context,
        solver->grids[0]->exitation_matrix->size_y,
        solver->grids[0]->exitation_matrix->size_x);

    // transpose measurment pattern
    error |= linalgcl_matrix_transpose(temp1, measurment_pattern,
        matrix_program, queue);

    // transpose exitation_matrix
    error |= linalgcl_matrix_transpose(temp2, solver->grids[0]->exitation_matrix,
        matrix_program, queue);

    // calc voltage_calculation matrix
    error |= linalgcl_matrix_multiply(solver->voltage_calculation, temp1, temp2,
        matrix_program, queue);

    // cleanup
    linalgcl_matrix_release(&measurment_pattern);
    linalgcl_matrix_release(&drive_pattern);
    linalgcl_matrix_release(&temp1);
    linalgcl_matrix_release(&temp2);

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
    if (solver->programs != NULL) {
        ert_solver_program_release(&solver->programs[0]);
        ert_solver_program_release(&solver->programs[1]);
        free(solver->programs);
    }

    if (solver->grids != NULL) {
        ert_grid_release(&solver->grids[0]);
        ert_grid_release(&solver->grids[1]);
        free(solver->grids);
    }

    if (solver->conjugate_solver != NULL) {
        ert_conjugate_solver_release(&solver->conjugate_solver[0]);
        ert_conjugate_solver_release(&solver->conjugate_solver[1]);
        free(solver->conjugate_solver);
    }

    ert_electrodes_release(&solver->electrodes);
    linalgcl_matrix_release(&solver->voltage_calculation);
    linalgcl_matrix_release(&solver->applied_phi);
    linalgcl_matrix_release(&solver->lead_phi);
    linalgcl_matrix_release(&solver->calculated_voltage);
    linalgcl_matrix_release(&solver->measured_voltage);

    if (solver->lead_f != NULL) {
        for (linalgcl_size_t i = 0; i < solver->measurment_count; i++) {
            linalgcl_matrix_release(&solver->lead_f[i]);
        }
        free(solver->lead_f);
    }

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
linalgcl_error_t ert_solver_copy_to_column(ert_solver_program_t program,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue) {
    // check input
    if ((program == NULL) || (matrix == NULL) || (vector == NULL) ||
        (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(program->kernel_copy_to_column,
        0, sizeof(cl_mem), &matrix->device_data);
    cl_error |= clSetKernelArg(program->kernel_copy_to_column,
        1, sizeof(cl_mem), &vector->device_data);
    cl_error |= clSetKernelArg(program->kernel_copy_to_column,
        2, sizeof(linalgcl_size_t), &column);
    cl_error |= clSetKernelArg(program->kernel_copy_to_column,
        3, sizeof(linalgcl_size_t), &matrix->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = vector->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, program->kernel_copy_to_column,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// copy from column
linalgcl_error_t ert_solver_copy_from_column(ert_solver_program_t program,
    linalgcl_matrix_t matrix, linalgcl_matrix_t vector, linalgcl_size_t column,
    cl_command_queue queue) {
    // check input
    if ((program == NULL) || (matrix == NULL) || (vector == NULL) ||
        (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(program->kernel_copy_from_column,
        0, sizeof(cl_mem), &matrix->device_data);
    cl_error |= clSetKernelArg(program->kernel_copy_from_column,
        1, sizeof(cl_mem), &vector->device_data);
    cl_error |= clSetKernelArg(program->kernel_copy_from_column,
        2, sizeof(linalgcl_size_t), &column);
    cl_error |= clSetKernelArg(program->kernel_copy_from_column,
        3, sizeof(linalgcl_size_t), &matrix->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = vector->size_x;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, program->kernel_copy_from_column,
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
        linalgcl_matrix_create(&solver->applied_f[i], context, solver->grids[0]->mesh->vertex_count, 1);

        // get current pattern
        ert_solver_copy_from_column(solver->programs[0], drive_pattern, current, i, queue);

        // calc f
        linalgcl_matrix_multiply(solver->applied_f[i], solver->grids[0]->exitation_matrix,
            current, matrix_program, queue);
    }

    // create measurment pattern
    for (linalgcl_size_t i = 0; i < solver->measurment_count; i++) {
        // create matrix
        linalgcl_matrix_create(&solver->lead_f[i], context, solver->grids[0]->mesh->vertex_count, 1);

        // get current pattern
        ert_solver_copy_from_column(solver->programs[0], measurment_pattern, current, i, queue);

        // calc f
        linalgcl_matrix_multiply(solver->lead_f[i], solver->grids[0]->exitation_matrix,
            current, matrix_program, queue);
    }

    // cleanup
    linalgcl_matrix_release(&current);

    return LINALGCL_SUCCESS;
}

/*// calc jacobian
linalgcl_error_t ert_solver_calc_jacobian(ert_solver_t solver,
    ert_solver_program_t program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(program->kernel_calc_jacobian,
        0, sizeof(cl_mem), &solver->jacobian->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        1, sizeof(cl_mem), &solver->applied_phi->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        2, sizeof(cl_mem), &solver->lead_phi->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        3, sizeof(cl_mem), &solver->grid->gradient_matrix_sparse->values->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        4, sizeof(cl_mem), &solver->grid->gradient_matrix_sparse->column_ids->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        5, sizeof(cl_mem), &solver->grid->area->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        6, sizeof(linalgcl_size_t), &solver->jacobian->size_y);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        7, sizeof(linalgcl_size_t), &solver->lead_phi->size_y);
    cl_error |= clSetKernelArg(program->kernel_calc_jacobian,
        8, sizeof(linalgcl_size_t), &solver->applied_phi->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global[2] = { solver->jacobian->size_x, solver->jacobian->size_y };
    size_t local[2] = { LINALGCL_BLOCK_SIZE, LINALGCL_BLOCK_SIZE };

    cl_error = clEnqueueNDRangeKernel(queue, program->kernel_calc_jacobian,
        2, NULL, global, local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}

// calc gradient
linalgcl_error_t ert_solver_calc_gradient(ert_solver_t solver,
    ert_solver_program_t program, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (program == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(program->kernel_calc_gradient,
        0, sizeof(cl_mem), &solver->gradient->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_gradient,
        1, sizeof(cl_mem), &solver->jacobian->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_gradient,
        2, sizeof(cl_mem), &solver->measured_voltage->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_gradient,
        3, sizeof(cl_mem), &solver->calculated_voltage->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_gradient,
        4, sizeof(cl_mem), &solver->sigma->device_data);
    cl_error |= clSetKernelArg(program->kernel_calc_gradient,
        5, sizeof(linalgcl_size_t), &solver->jacobian->size_x);
    cl_error |= clSetKernelArg(program->kernel_calc_gradient,
        6, sizeof(linalgcl_size_t), &solver->jacobian->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global = solver->jacobian->size_y;
    size_t local = LINALGCL_BLOCK_SIZE;

    cl_error = clEnqueueNDRangeKernel(queue, program->kernel_calc_gradient,
        1, NULL, &global, &local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}*/

// forward solving
actor_error_t ert_solver_forward(actor_process_t self, ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (matrix_program == NULL) || (context == NULL) ||
        (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // message
    actor_message_t message = NULL;

    // frame counter
    linalgcl_size_t frames = 0;;

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // create phi buffer
    linalgcl_matrix_t phi = NULL;
    error = linalgcl_matrix_create(&phi, context, solver->applied_phi->size_x, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // solve forever
    while (1) {
        // increment frame counter
        frames++;

        // solve drive patterns
        for (linalgcl_size_t i = 0; i < solver->drive_count; i++) {
            // copy current applied phi to vector
            error = ert_solver_copy_from_column(solver->programs[0], solver->applied_phi,
                phi, i, queue);

            // solve for phi
            error |= ert_conjugate_solver_solve(solver->conjugate_solver[0],
                phi, solver->applied_f[i], 10, matrix_program, queue);

            // copy vector to applied phi
            error |= ert_solver_copy_to_column(solver->programs[0], solver->applied_phi,
                phi, i, queue);

            // check success
            if (error != LINALGCL_SUCCESS) {
                // cleanup
                linalgcl_matrix_release(&phi);

                return ACTOR_ERROR;
            }
        }

        if (self == NULL) {
            break;
        }

        // receive stop message
        if (actor_receive(self, &message, 0.0) == ACTOR_ERROR_TIMEOUT) {
            continue;
        }

        // check for end message
        if (message->type == ACTOR_TYPE_ERROR_MESSAGE) {
            // cleanup
            actor_message_release(&message);

            break;
        }

        // cleanup
        actor_message_release(&message);
    }

    // cleanup
    linalgcl_matrix_release(&phi);

    // calc voltage
    error = linalgcl_matrix_multiply(solver->calculated_voltage, solver->voltage_calculation,
        solver->applied_phi, matrix_program, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // get end time
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // print frames per second
    printf("Forward: frames per second: %f\n", (linalgcl_matrix_data_t)frames / (end - start));

    return ACTOR_SUCCESS;
}

// inverse solving
actor_error_t ert_solver_inverse(actor_process_t self, ert_solver_t solver,
    linalgcl_matrix_program_t matrix_program, cl_context context,
    cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (matrix_program == NULL) || (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // message
    actor_message_t message = NULL;

    // frame counter
    linalgcl_size_t frames = 0;;

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // create phi buffer
    linalgcl_matrix_t phi = NULL;
    error = linalgcl_matrix_create(&phi, context, solver->lead_phi->size_x, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // solve forever
    while (1) {
        // increment frame counter
        frames++;

        // solve measurment patterns
        for (linalgcl_size_t i = 0; i < solver->measurment_count; i++) {
            // copy current applied phi to vector
            error = ert_solver_copy_from_column(solver->programs[1], solver->lead_phi,
                phi, i, queue);

            // solve for phi
            error |= ert_conjugate_solver_solve(solver->conjugate_solver[1],
                phi, solver->lead_f[i], 10, matrix_program, queue);

            // copy vector to applied phi
            error |= ert_solver_copy_to_column(solver->programs[1], solver->lead_phi,
                phi, i, queue);

            // check success
            if (error != LINALGCL_SUCCESS) {
                // cleanup
                linalgcl_matrix_release(&phi);

                return ACTOR_ERROR;
            }
        }

        // calc jacobian
        // ert_solver_calc_jacobian(solver, solver->program1, queue);

        /*// calc gradient
        ert_solver_calc_gradient(solver, solver->program1, queue);

        // add to sigma
        linalgcl_matrix_scalar_multiply(solver->gradient, solver->gradient,
            0.0001f, matrix_program, queue);
        linalgcl_matrix_add(solver->sigma, solver->sigma, solver->gradient,
            matrix_program, queue);
        ert_grid_update_system_matrix(solver->grid, queue);*/

        // receive stop message
        if (actor_receive(self, &message, 0.0) == ACTOR_ERROR_TIMEOUT) {
            continue;
        }

        // check for end message
        if (message->type == ACTOR_TYPE_ERROR_MESSAGE) {
            // cleanup
            actor_message_release(&message);

            break;
        }

        // cleanup
        actor_message_release(&message);
    }

    // cleanup
    linalgcl_matrix_release(&phi);

    // get end time
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // print frames per second
    printf("Inverse: frames per second: %f\n", (double)frames / (end - start));

    return ACTOR_SUCCESS;
}
