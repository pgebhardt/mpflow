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
    solver->grid_program = NULL;
    solver->gradient_solver = NULL;

    // load program
    error = ert_grid_program_create(&solver->grid_program, context, device_id,
        "src/grid.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // create grid memory
    solver->grids = malloc(sizeof(ert_grid_s) * solver->max_grids);

    // check success
    if (solver->grids == NULL) {
        // cleanup
        ert_solver_release(&solver);

        return LINALGCL_ERROR;
    }

    // create gradient memory
    solver->gradient_solver = malloc(sizeof(ert_gradient_solver_s) * solver->max_grids);

    // check success
    if (solver->gradient_solver == NULL) {
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
    ert_grid_program_release(&solver->grid_program);

    // release grids
    for (linalgcl_size_t i = 0; i < solver->grid_count; i++) {
        ert_grid_release(&solver->grids[i]);
    }
    free(solver->grids);

    // release gradient solver
    for(linalgcl_size_t i = 0; i < solver->grid_count; i++) {
        ert_gradient_solver_release(&solver->gradient_solver[i]);
    }
    free(solver->gradient_solver);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALGCL_SUCCESS;
}

// add coarser grid
linalgcl_error_t ert_solver_add_coarser_grid(ert_solver_t solver,
    ert_mesh_t mesh, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_device_id device_id, cl_command_queue queue) {
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
    error = ert_grid_create(&solver->grids[solver->grid_count],
        matrix_program, mesh, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // add new gradient_solver
    error = ert_gradient_solver_create(&solver->gradient_solver[solver->grid_count],
        mesh->vertex_count, matrix_program, context, device_id, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // regularize system matrix
    linalgcl_sparse_matrix_unfold(solver->gradient_solver[solver->grid_count]->system_matrix,
        solver->grids[solver->grid_count]->system_matrix, matrix_program, queue);
    ert_gradient_solver_regularize_system_matrix(solver->gradient_solver[solver->grid_count],
        0.0, matrix_program, queue);

    // increment grid counter
    solver->grid_count++;

    return LINALGCL_SUCCESS;
}

// do Multigrid step
linalgcl_error_t ert_solver_multigrid(ert_solver_t solver, linalgcl_size_t n, linalgcl_size_t depth,
    linalgcl_matrix_t f, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_command_queue queue) {
    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    if (n >= depth - 1) {
        // regularize f
        linalgcl_sparse_matrix_vector_multiply(solver->gradient_solver[n]->temp_vector,
            solver->grids[n]->system_matrix, f, matrix_program, queue);

        // calc error
        error = ert_gradient_solver_solve(solver->gradient_solver[n], solver->grids[n]->x, f, matrix_program,
            queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("conjugate gradient error!\n");
            return error;
        }

    }
    else {
        // calc residuum r = f - A * x
        error  = linalgcl_sparse_matrix_vector_multiply(solver->grids[n]->temp1,
            solver->grids[n]->system_matrix, solver->grids[n]->x, matrix_program, queue);
        error += linalgcl_matrix_scalar_multiply(solver->grids[n]->temp2,
            solver->grids[n]->temp1, -1.0, matrix_program, queue);
        error += linalgcl_matrix_add(solver->grids[n]->residuum, f,
            solver->grids[n]->temp2, matrix_program, queue);

        // pre smoothing
        linalgcl_sparse_matrix_vector_multiply(solver->grids[n]->temp1, solver->grids[n]->smooth_phi,
            solver->grids[n]->residuum, matrix_program, queue);
        linalgcl_sparse_matrix_vector_multiply(solver->grids[n]->residuum, solver->grids[n]->smooth_phi,
            solver->grids[n]->temp1, matrix_program, queue);

        // calc coarser residuum 
        error = linalgcl_sparse_matrix_vector_multiply(solver->grids[n + 1]->f,
            solver->grids[n]->restrict_phi, solver->grids[n]->residuum, matrix_program, queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("calc coarser residuum error!\n");
            return error;
        }

        // calc error
        error = ert_solver_multigrid(solver, n + 1, depth, solver->grids[n + 1]->f, matrix_program, context, queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("calc error error!\n");
            return error;
        }

        // prolongate error
        error = linalgcl_sparse_matrix_vector_multiply(solver->grids[n]->error,
            solver->grids[n + 1]->prolongate_phi, solver->grids[n + 1]->x, matrix_program, queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("prolongation error!\n");
            return error;
        }

        // post smoothing
        linalgcl_sparse_matrix_vector_multiply(solver->grids[n]->temp1, solver->grids[n]->smooth_phi,
            solver->grids[n]->error, matrix_program, queue);
        linalgcl_sparse_matrix_vector_multiply(solver->grids[n]->error, solver->grids[n]->smooth_phi,
            solver->grids[n]->temp1, matrix_program, queue);

        // update x
        error = linalgcl_matrix_add(solver->grids[n]->x, solver->grids[n]->x, solver->grids[n]->error,
            matrix_program, queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("update x error!\n");
            return error;
        }
    }

    return LINALGCL_SUCCESS;
}

// do v cycle
linalgcl_error_t ert_solver_solve(ert_solver_t solver, linalgcl_matrix_t x,
    linalgcl_matrix_t f, linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_command_queue queue) {
    // check input
    if ((solver == NULL) || (x == NULL) || (matrix_program == NULL) ||
        (f == NULL) || (context == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_ERROR;

    // create matrices
    linalgcl_matrix_t temp1, temp2;
    linalgcl_matrix_t rh = solver->grids[0]->residuum;
    linalgcl_matrix_t rH = solver->grids[1]->residuum;
    linalgcl_matrix_t eh = solver->grids[0]->error;
    linalgcl_matrix_t eH = solver->grids[1]->error;
    error = linalgcl_matrix_create(&temp1, context, solver->grids[0]->mesh->vertex_count, 1);
    error += linalgcl_matrix_create(&temp2, context, solver->grids[0]->mesh->vertex_count, 1);

    linalgcl_matrix_copy(solver->grids[0]->x, x, queue, CL_TRUE);

    // v cycle
    linalgcl_size_t cycles = 5;
    linalgcl_size_t depth = solver->grid_count;

    for (linalgcl_size_t j = 0; j < 5; j++) {
        for (linalgcl_size_t i = 0; i < cycles; i++) {
            // calc depth
            depth = cycles - i > solver->grid_count ? solver->grid_count : cycles - i;
            printf("cycle %d: depth %d\n", i, depth);

            // do Multigrid step
            ert_solver_multigrid(solver, 0, depth,
                f, matrix_program, context, queue);
        }
    }

    // cleanup
    linalgcl_matrix_release(&temp1);
    linalgcl_matrix_release(&temp2);

    return LINALGCL_SUCCESS;
}
