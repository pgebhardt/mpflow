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

#include <actor/actor.h>
#include <linalgcl/linalgcl.h>
#include "mesh.h"
#include "basis.h"
#include "image.h"
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

static actor_process_function_t main_process = ^(actor_process_t self) {
    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;
    cl_int cl_error = CL_SUCCESS;

    // Connect to a compute device
    cl_device_id device_id;
    cl_error = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // Create a compute context 
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // Create a command commands
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &cl_error);

    if (cl_error != CL_SUCCESS) {
        // cleanup
        clReleaseContext(context);

        return ACTOR_ERROR;
    }

    // create matrix program
    linalgcl_matrix_program_t program = NULL;
    error = linalgcl_matrix_create_programm(&program, context, device_id,
        "/usr/local/include/linalgcl/matrix.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        clReleaseContext(context);

        return ACTOR_ERROR;
    }

    // create mesh
    ert_mesh_t mesh[2];
    error  = ert_mesh_create(&mesh[0], 1.0, 1.0 / 16.0, context);
    error += ert_mesh_create(&mesh[1], 1.0, 1.0 / 8.0, context);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        clReleaseContext(context);
        clReleaseCommandQueue(queue);

        return ACTOR_ERROR;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(mesh[0]->vertices, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh[0]->elements, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh[1]->vertices, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh[1]->elements, queue, CL_TRUE);
    printf("vertices: %d\n", mesh[0]->vertex_count);

    // create solver
    ert_solver_t solver;
    error = ert_solver_create(&solver, 2, context, device_id);

    error += ert_solver_add_coarser_grid(solver, mesh[0], program,
        context, queue);
    error += ert_solver_add_coarser_grid(solver, mesh[1], program,
        context, queue);
    clFinish(queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        clReleaseCommandQueue(queue);
        clReleaseContext(context);

        return ACTOR_ERROR;
    }

    // calc intergrid transfer matrices
    error  = ert_grid_init_intergrid_transfer_matrices(solver->grids[0], NULL,
        solver->grids[1], program, context, queue);
    error += ert_grid_init_intergrid_transfer_matrices(solver->grids[1],
        solver->grids[0], NULL, program, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        printf("intergrid transfer ging nicht!\n");

        return ACTOR_ERROR;
    }

    linalgcl_matrix_t x;
    linalgcl_matrix_create(&x, context, mesh[0]->vertex_count, 1);

    for (linalgcl_size_t i = 1; i < mesh[0]->vertex_count; i++) {
        x->host_data[i] = 1.0;
    }

    linalgcl_matrix_copy_to_device(x, queue, CL_TRUE);

    linalgcl_matrix_t j, f;
    linalgcl_matrix_create(&j, context, 10, 1);
    linalgcl_matrix_create(&f, context, mesh[0]->vertex_count, 1);

    // set j
    linalgcl_matrix_set_element(j, 1.0, 1, 0);
    linalgcl_matrix_set_element(j, -1.0, 2, 0);
    linalgcl_matrix_copy_to_device(j, queue, CL_TRUE);

    // calc f matrix
    // f = B * j
    linalgcl_matrix_multiply(program, queue, f,
        solver->grids[0]->exitation_matrix, j);
    linalgcl_matrix_release(&j);

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    clFinish(queue);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    error = ert_solver_v_cycle(solver, x, f, program, context, queue);
    // error = ert_solver_conjugate_gradient(solver->grids[0], x, f, program, context, queue);
    clFinish(queue);

    // get end time
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;
    printf("Solving time: %f\n", end - start);

    if (error != LINALGCL_SUCCESS) {
        printf("Multigrid geht nicht!\n");
        return ACTOR_ERROR;
    }

    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh[0], context, device_id);
    linalgcl_matrix_copy_to_device(image->elements, queue, CL_TRUE);

    ert_image_calc(image, x, queue);
    clFinish(queue);
    linalgcl_matrix_copy_to_host(image->image, queue, CL_TRUE);
    linalgcl_matrix_save("image.txt", image->image);
    system("python src/script.py");

    // cleanup
    linalgcl_matrix_release(&x);
    linalgcl_matrix_release(&f);
    ert_solver_release(&solver);
    ert_image_release(&image);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return ACTOR_SUCCESS;
};

int main(int argc, char* argv[]) {
    // error
    actor_error_t error = ACTOR_SUCCESS;

    // create node
    actor_node_t node = NULL;
    error = actor_node_create(&node, 0, 100);

    // check success
    if (error != ACTOR_SUCCESS) {
        return EXIT_FAILURE;
    }

    // start main process
    error = actor_spawn(node, NULL, main_process);

    // check success
    if (error != ACTOR_SUCCESS) {
        // cleanup
        actor_node_release(&node);

        return EXIT_FAILURE;
    }

    // wait for processes to complete
    while (actor_node_wait_for_processes(node, 10.0) != ACTOR_SUCCESS) {
        // wait
    }

    // cleanup
    actor_node_release(&node);

    return EXIT_SUCCESS;
}
