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
    error = ert_grid_create(&solver->grids[solver->grid_count],
        matrix_program, mesh, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // increment grid counter
    solver->grid_count++;

    return LINALGCL_SUCCESS;
}

// solve conjugate gradient
linalgcl_error_t ert_solver_conjugate_gradient(ert_grid_t grid,
    linalgcl_matrix_t initial_guess, linalgcl_matrix_t f,
    linalgcl_matrix_program_t matrix_program,
    cl_context context, cl_command_queue queue) {
    // check input
    if ((grid == NULL) || (initial_guess == NULL) || (f == NULL) || (queue == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    linalgcl_matrix_t x, r, p, temp1, temp2;
    linalgcl_matrix_data_t alpha, beta, temp3, rnorm;

    // create matrices
    error  = linalgcl_matrix_create(&x, context, initial_guess->size_x, 1);
    error += linalgcl_matrix_create(&r, context, initial_guess->size_x, 1);
    error += linalgcl_matrix_create(&p, context, initial_guess->size_x, 1);
    error += linalgcl_matrix_create(&temp1, context, initial_guess->size_x, 1);
    error += linalgcl_matrix_create(&temp2, context, initial_guess->size_x, 1);
    // set data
    // set x0 to initial_guess
    error += linalgcl_matrix_copy(matrix_program, queue, x, initial_guess);

    // calc r0 = f - A * x0
    error += linalgcl_matrix_scalar_multiply(matrix_program, queue, temp1, x, -1.0);

    error += linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, temp2,
        grid->system_matrix, temp1);

    error += linalgcl_matrix_add(matrix_program, queue, r, f, temp2);
    clFinish(queue);

    // init p0
    error += linalgcl_matrix_copy(matrix_program, queue, p, r);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // iteration
    for (linalgcl_size_t i = 0; i < 5; i++) {
        // calc alpha
        // calc A * p
        linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, temp1,
            grid->system_matrix, p);
        clFinish(queue);

        // copy to host
        linalgcl_matrix_copy_to_host(r, queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(p, queue, CL_TRUE);
        linalgcl_matrix_copy_to_host(temp1, queue, CL_TRUE);

        // r * r
        rnorm = 0.0;
        temp3 = 0.0;
        for (linalgcl_size_t k = 0; k < r->size_x; k++) {
            rnorm += r->host_data[k] * r->host_data[k];
            temp3 += p->host_data[k] * temp1->host_data[k];
        }
        alpha = rnorm / temp3;

        // calc xi = xi-1 + alpha * pi-1
        linalgcl_matrix_scalar_multiply(matrix_program, queue, temp2,
            p, alpha);
        error = linalgcl_matrix_add(matrix_program, queue, x, x, temp2);

        // check success
        if (error != LINALGCL_SUCCESS) {
            return LINALGCL_ERROR;
        }

        // calc ri = ri-1 - alpha * A * pi-1
        linalgcl_matrix_scalar_multiply(matrix_program, queue, temp2, temp1, -alpha);
        linalgcl_matrix_add(matrix_program, queue, r, r, temp2);
        clFinish(queue);

        // calc beta
        linalgcl_matrix_copy_to_host(r, queue, CL_TRUE);

        beta = 0.0;
        for (linalgcl_size_t k = 0; k < r->size_x; k++) {
            beta += r->host_data[k] * r ->host_data[k];
        }
        beta /= rnorm;

        // calc new pi = ri + beta * pi-1
        linalgcl_matrix_scalar_multiply(matrix_program, queue, temp1, p, beta);
        linalgcl_matrix_add(matrix_program, queue, p, r, temp1);
        clFinish(queue);
    }

    clFinish(queue);
    linalgcl_matrix_copy(matrix_program, queue, initial_guess, x);

    // cleanup
    linalgcl_matrix_release(&x);
    linalgcl_matrix_release(&r);
    linalgcl_matrix_release(&p);
    linalgcl_matrix_release(&temp1);
    linalgcl_matrix_release(&temp2);

    return LINALGCL_SUCCESS;
}

// do v cycle
linalgcl_error_t ert_solver_v_cycle(ert_solver_t solver, linalgcl_matrix_t x,
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
    linalgcl_matrix_t rh, rH, eh, eH, temp1, temp2;
    error  = linalgcl_matrix_create(&rh, context, solver->grids[0]->mesh->vertex_count, 1);
    error += linalgcl_matrix_create(&rH, context, solver->grids[1]->mesh->vertex_count, 1);
    error += linalgcl_matrix_create(&eh, context, solver->grids[0]->mesh->vertex_count, 1);
    error += linalgcl_matrix_create(&eH, context, solver->grids[1]->mesh->vertex_count, 1);
    error += linalgcl_matrix_create(&temp1, context, solver->grids[0]->mesh->vertex_count, 1);
    error += linalgcl_matrix_create(&temp2, context, solver->grids[0]->mesh->vertex_count, 1);

    // v cycle
    for (linalgcl_size_t i = 0; i < 5; i++) {
        // set residuum rh = fh - Ah * xh
        error  = linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, temp1,
            solver->grids[0]->system_matrix, x);
        error += linalgcl_matrix_scalar_multiply(matrix_program, queue, temp2, temp1, -1.0);
        error += linalgcl_matrix_add(matrix_program, queue, rh, f, temp2);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("calc residuum error!\n");
            return error;
        }

        // calc coarser residuum rH = IhH * rh
        error = linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, rH,
            solver->grids[0]->restrict_phi, rh);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("restriction error!\n");
            return error;
        }

        // calc error
        error = ert_solver_conjugate_gradient(solver->grids[1], eH, rH, matrix_program, context, queue);
        clFinish(queue);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("conjugate gradient error!\n");
            return error;
        }

        // prolongate error
        error = linalgcl_sparse_matrix_vector_multiply(matrix_program, queue, eh,
            solver->grids[1]->prolongate_phi, eH);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("prolongation error!\n");
            return error;
        }

        // update x
        error = linalgcl_matrix_add(matrix_program, queue, x, x, eh);

        // check success
        if (error != LINALGCL_SUCCESS) {
            printf("update x error!\n");
            return error;
        }
    }

    // cleanup
    linalgcl_matrix_release(&rh);
    linalgcl_matrix_release(&rH);
    linalgcl_matrix_release(&eh);
    linalgcl_matrix_release(&eH);
    linalgcl_matrix_release(&temp1);
    linalgcl_matrix_release(&temp2);

    return LINALGCL_SUCCESS;
}
