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
#include "solver.h"

// create solver
linalgcl_error_t ert_solver_create(ert_solver_t* solverPointer,
    linalgcl_size_t max_grids, cl_context context, cl_device_id device_id) {
    // check input
    if ((solverPointer == NULL) || (context == NULL)) {
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
    solver->grids = NULL;
    solver->grid_count = 0;
    solver->max_grids = max_grids;
    solver->program = NULL;

    // load program
    error = ert_solver_program_create(&solver->program, context, device_id,
        "src/solver.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // create grid memory
    solver->grids = malloc(sizeof(ert_solver_grid_s) * solver->max_grids);

    // check success
    if (solver->grids == NULL) {
        // cleanup
        ert_solver_release(&solver);

        return LINALGCL_ERROR;
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

    // release grids
    for (linalgcl_size_t i = 0; i < solver->grid_count; i++) {
        ert_solver_grid_release(&solver->grids[i]);
    }
    free(solver->grids);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// add coarser grid
linalgcl_error_t ert_solver_add_coarser_grid(ert_solver_t solver,
    ert_mesh_t mesh, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (mesh == NULL) || (matrix_program == NULL) ||
        (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // check grid count
    if (solver->grid_count == solver->max_grids) {
        return LINALGCL_ERROR;
    }

    // add new grid
    error = ert_solver_grid_create(&solver->grids[solver->grid_count],
        solver, matrix_program, mesh, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // increment grid counter
    solver->grid_count++;

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

// create solver grid
linalgcl_error_t ert_solver_grid_create(ert_solver_grid_t* gridPointer,
    ert_solver_t solver, linalgcl_matrix_program_t matrix_program,
    ert_mesh_t mesh, cl_context context, cl_command_queue queue) {
    // check input
    if ((gridPointer == NULL) || (solver == NULL) || (mesh == NULL) ||
        (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init grid pointer
    *gridPointer = NULL;

    // create grid struct
    ert_solver_grid_t grid = malloc(sizeof(ert_solver_grid_s));

    // check success
    if (grid == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    grid->mesh = mesh;
    grid->system_matrix = NULL;
    grid->gradient_matrix_sparse = NULL;
    grid->gradient_matrix = NULL;
    grid->sigma = NULL;
    grid->area = NULL;

    // create sigma and area vector
    error  = linalgcl_matrix_create(&grid->sigma, context,
        grid->mesh->element_count, 1);
    error += linalgcl_matrix_create(&grid->area, context,
        grid->mesh->element_count, 1);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_grid_release(&grid);

        return error;
    }

    // init to uniform sigma
    for (linalgcl_size_t i = 0; i < grid->sigma->size_x; i++) {
        if (i < grid->mesh->element_count) {
            linalgcl_matrix_set_element(grid->sigma, 1.0, i, 0);
        }
        else {
            linalgcl_matrix_set_element(grid->sigma, 0.0, i, 0);
        }
    }

    // calc initial system matrix
    // create matrices
    linalgcl_matrix_t system_matrix, gradient_matrix, sigma_matrix;
    error = linalgcl_matrix_create(&system_matrix, context,
        grid->mesh->vertex_count, grid->mesh->vertex_count);
    error += linalgcl_matrix_create(&gradient_matrix, context,
        2 * grid->mesh->element_count, grid->mesh->vertex_count);
    error += linalgcl_matrix_create(&grid->gradient_matrix, context,
        grid->mesh->vertex_count, 2 * grid->mesh->element_count);
    error += linalgcl_matrix_unity(&sigma_matrix, context,
        2 * grid->mesh->element_count);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_grid_release(&grid);

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

    linalgcl_matrix_data_t s, a, b, c, area;

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
        a = sqrt((x[1] - x[2]) * (x[1] - x[2]) + (y[1] - y[2]) * (y[1] - y[2]));
        b = sqrt((x[2] - x[0]) * (x[2] - x[0]) + (y[2] - y[0]) * (y[2] - y[0]));
        c = sqrt((x[0] - x[1]) * (x[0] - x[1]) + (y[0] - y[1]) * (y[0] - y[1]));
        s = (a + b + c) / 2.0;
        area = sqrt(s * (s - a) * (s - b) * (s - c));

        linalgcl_matrix_set_element(grid->area, area, k, 0);
        linalgcl_matrix_set_element(sigma_matrix, area, 2 * k, 2 * k);
        linalgcl_matrix_set_element(sigma_matrix, area, 2 * k + 1, 2 * k + 1);

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
        ert_solver_grid_release(&grid);

        return LINALGCL_ERROR;
    }

    // calc system matrix
    linalgcl_matrix_t temp = NULL;
    error = linalgcl_matrix_create(&temp, context, grid->mesh->vertex_count,
        2 * grid->mesh->element_count);
    error += linalgcl_matrix_multiply(matrix_program, queue, temp,
        grid->gradient_matrix, sigma_matrix);
    error += linalgcl_matrix_multiply(matrix_program, queue, system_matrix,
        temp, gradient_matrix);

    // cleanup
    linalgcl_matrix_release(&sigma_matrix);
    linalgcl_matrix_release(&gradient_matrix);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        linalgcl_matrix_release(&system_matrix);
        ert_solver_grid_release(&grid);

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
        ert_solver_grid_release(&grid);

        return LINALGCL_ERROR;
    }

    // set grid pointer
    *gridPointer = grid;

    return LINALGCL_SUCCESS;
}

// release solver grid
linalgcl_error_t ert_solver_grid_release(ert_solver_grid_t* gridPointer) {
    // check input
    if ((gridPointer == NULL) || (*gridPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get grid
    ert_solver_grid_t grid = *gridPointer;

    // cleanup
    ert_mesh_release(&grid->mesh);
    linalgcl_sparse_matrix_release(&grid->system_matrix);
    linalgcl_matrix_release(&grid->gradient_matrix);
    linalgcl_sparse_matrix_release(&grid->gradient_matrix_sparse);
    linalgcl_matrix_release(&grid->sigma);
    linalgcl_matrix_release(&grid->area);

    // free struct
    free(grid);

    // set grid pointer to NULL
    *gridPointer = NULL;

    return LINALGCL_SUCCESS;
}

// update system matrix
linalgcl_error_t ert_solver_update_system_matrix(ert_solver_grid_t grid,
    ert_solver_t solver, cl_command_queue queue) {
    // check input
    if ((grid == NULL) || (solver == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    cl_int cl_error = CL_SUCCESS;

    // set kernel arguments
    cl_error  = clSetKernelArg(solver->program->kernel_update_system_matrix,
        0, sizeof(cl_mem), &grid->system_matrix->values->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        1, sizeof(cl_mem), &grid->system_matrix->column_ids->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        2, sizeof(cl_mem), &grid->gradient_matrix_sparse->values->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        3, sizeof(cl_mem), &grid->gradient_matrix_sparse->column_ids->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        4, sizeof(cl_mem), &grid->gradient_matrix->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        5, sizeof(cl_mem), &grid->sigma->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        6, sizeof(cl_mem), &grid->area->device_data);
    cl_error += clSetKernelArg(solver->program->kernel_update_system_matrix,
        7, sizeof(unsigned int), &grid->gradient_matrix->size_y);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    // execute kernel_update_system_matrix
    size_t global[2] = { grid->system_matrix->size_x, LINALGCL_BLOCK_SIZE };
    size_t local[2] = { LINALGCL_BLOCK_SIZE, LINALGCL_BLOCK_SIZE };

    cl_error = clEnqueueNDRangeKernel(queue,
        solver->program->kernel_update_system_matrix, 2,
        NULL, global, local, 0, NULL, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return LINALGCL_ERROR;
    }

    return LINALGCL_SUCCESS;
}
