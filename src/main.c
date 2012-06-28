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
        return ACTOR_ERROR;
    }

    // create matrix program
    linalgcl_matrix_program_t program = NULL;
    error = linalgcl_matrix_create_programm(&program, context, device_id,
        "/usr/local/include/linalgcl/matrix.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create mesh
    ert_mesh_t mesh;
    error  = ert_mesh_create(&mesh, 1.0, 1.0 / 16.0, context);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(mesh->vertices, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh->elements, queue, CL_TRUE);

    // create grid
    ert_grid_t grid;
    error = ert_grid_create(&grid, program, mesh, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create image
    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh, context, device_id);
    linalgcl_matrix_copy_to_device(image->elements, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(image->image, queue, CL_TRUE);

    // create solver
    ert_minres_solver_t solver;
    error = ert_minres_solver_create(&solver, grid, context, device_id, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // x vector
    linalgcl_matrix_t x;
    linalgcl_matrix_create(&x, context, mesh->vertex_count, 1);
    linalgcl_matrix_copy_to_device(x, queue, CL_TRUE);

    // right hand side
    linalgcl_matrix_t j, f;
    linalgcl_matrix_create(&j, context, 10, 1);
    linalgcl_matrix_create(&f, context, mesh->vertex_count, 1);

    // set j
    linalgcl_matrix_set_element(j, 1.0, 1, 0);
    linalgcl_matrix_set_element(j, -1.0, 5, 0);
    linalgcl_matrix_copy_to_device(j, queue, CL_TRUE);

    // calc f matrix
    // f = B * j
    linalgcl_matrix_multiply(f, grid->exitation_matrix, j, program, queue);
    clFinish(queue);
    linalgcl_matrix_release(&j);
    linalgcl_matrix_copy_to_host(f, queue, CL_TRUE);
    linalgcl_matrix_save("f.txt", f);

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    clFinish(queue);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // solve
    error = ert_minres_solver_solve(solver, x, f, program, queue);

    printf("success: %d\n", error);

    // get end time
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;
    printf("Solving time: %f\n", end - start);

    ert_image_calc(image, x, queue);
    clFinish(queue);
    linalgcl_matrix_copy_to_host(image->image, queue, CL_TRUE);
    linalgcl_matrix_save("image.txt", image->image);
    system("python src/script.py");

    // cleanup
    linalgcl_matrix_release(&x);
    linalgcl_matrix_release(&f);
    ert_minres_solver_release(&solver);
    ert_grid_release(&grid);
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
