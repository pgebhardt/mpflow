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

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <actor/actor.h>
#include <linalgcl/linalgcl.h>
#include "mesh.h"

void print_matrix(linalgcl_matrix_t matrix) {
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

    // create mesh
    ert_mesh_t mesh = NULL;
    error = ert_mesh_create(&mesh, 1.0, 1.0 / 16.0, context);

    // check success
    if (error != LINALGCL_SUCCESS) {
        // cleanup
        clReleaseContext(context);
        clReleaseCommandQueue(queue);

        return ACTOR_ERROR;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(mesh->vertices, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh->elements, queue, CL_TRUE);

    // save vertices
    linalgcl_matrix_save("vertices.txt", mesh->vertices);

    // save elements
    linalgcl_matrix_save("elements.txt", mesh->elements);

    // show grid
    system("python src/script.py");

    // cleanup
    ert_mesh_release(&mesh);
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
