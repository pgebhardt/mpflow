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

#include <linalgcl/linalgcl.h>
#include <actor/actor.h>
#include "mesh.h"
#include "basis.h"
#include "image.h"
#include "electrodes.h"
#include "grid.h"
#include "forward.h"
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

// forward solving process
actor_error_t forward_process(actor_process_t self, ert_solver_t solver,
    linalgcl_matrix_program_t program, cl_command_queue queue) {
    // error
    actor_error_t error = ACTOR_SUCCESS;

    // message
    actor_message_t message = NULL;

    // frame counter
    linalgcl_size_t frames = 0;;

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // run loop
    while (1) {
        // increment frame counter
        frames++;

        // solve forward
        ert_solver_forward_solve(solver, program, queue);

        // receive message
        error = actor_receive(self, &message, 0.0);

        // check for timeout
        if (error == ACTOR_ERROR_TIMEOUT) {
            continue;
        }

        // check for end of main process
        if (message->type == ACTOR_TYPE_ERROR_MESSAGE) {
            // cleanup
            actor_message_release(&message);

            break;
        }

        // cleanup
        actor_message_release(&message);
    }

    // get end time
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // print frames per second
    printf("Frames per second: %f\n", (double)frames / (end - start));

    return ACTOR_SUCCESS;
}

// main process
actor_error_t main_process(actor_process_t main) {
    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;
    cl_int cl_error = CL_SUCCESS;

    // Get Platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // Connect to a compute device
    cl_device_id device_id[2];
    cl_error = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 2, device_id, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // Create a compute context 
    cl_context context = clCreateContext(0, 2, device_id, NULL, NULL, &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // Create a command commands
    cl_command_queue queue0 = clCreateCommandQueue(context, device_id[0], 0, &cl_error);
    cl_command_queue queue1 = clCreateCommandQueue(context, device_id[1], 0, &cl_error);

    if (cl_error != CL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create matrix program
    linalgcl_matrix_program_t program = NULL;
    error = linalgcl_matrix_create_programm(&program, context, device_id[0],
        "/usr/local/include/linalgcl/matrix.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create mesh
    ert_mesh_t mesh;
    error = ert_mesh_create(&mesh, 0.045, 0.045 / 16.0, context);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(mesh->vertices, queue0, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh->elements, queue0, CL_TRUE);

    // create electrodes
    ert_electrodes_t electrodes;
    error = ert_electrodes_create(&electrodes, 36, 0.005, mesh);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create solver
    ert_solver_t solver;
    error = ert_solver_create(&solver, mesh, electrodes, 9, 18,
        program, context, device_id[0], queue0);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // Create image
    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh, context, device_id[0]);
    linalgcl_matrix_copy_to_device(image->elements, queue0, CL_FALSE);
    linalgcl_matrix_copy_to_device(image->image, queue0, CL_TRUE);

    // set sigma
    for (linalgcl_size_t i = 0; i < mesh->element_count; i++) {
        linalgcl_matrix_set_element(solver->sigma, 10.0f * 1E-3, i, 0);
    }
    linalgcl_matrix_copy_to_device(solver->sigma, queue0, CL_TRUE);
    ert_grid_update_system_matrix(solver->grid, queue0);
    clFinish(queue0);

    // start forward solving process
    actor_process_id_t forward;
    actor_spawn(main->node, &forward, ^actor_error_t(actor_process_t self) {
        // link to main process
        actor_process_link(self, main->nid, main->pid);

        return forward_process(self, solver, program, queue0);
    });
    printf("Forward solving process started!\n");

    // change sigma slowly
    for (linalgcl_size_t k = 0; k < 5; k++) {
        for (linalgcl_size_t i = 1; i <= 40; i++) {
            // set sigma
            for (linalgcl_size_t j = 0; j < mesh->element_count / 2; j++) {
                linalgcl_matrix_set_element(solver->sigma, (10.0f + (linalgcl_matrix_data_t)i * 1.0f) * 1E-3, j, 0);
            }
            linalgcl_matrix_copy_to_device(solver->sigma, queue1, CL_TRUE);
            ert_grid_update_system_matrix(solver->grid, queue1);

            // sleep a bit
            actor_process_sleep(main, 0.04);
        }
        for (int i = 39; i >= 0; i--) {
            // set sigma
            for (linalgcl_size_t j = 0; j < mesh->element_count / 2; j++) {
                linalgcl_matrix_set_element(solver->sigma, (10.0f + (linalgcl_matrix_data_t)i * 1.0f) * 1E-3, j, 0);
            }
            linalgcl_matrix_copy_to_device(solver->sigma, queue1, CL_TRUE);
            ert_grid_update_system_matrix(solver->grid, queue1);

            // sleep a bit
            actor_process_sleep(main, 0.04);
        }
    }

    // sleep a bit
    actor_process_sleep(main, 0.5);

    // stop forward process
    actor_send(main, main->nid, forward, ACTOR_TYPE_ERROR_MESSAGE,
        &forward, sizeof(actor_process_id_t));
    printf("Send stop signal to forward solving process!\n");

    // wait for forward process to stop
    actor_message_t message = NULL;
    actor_receive(main, &message, 10.0);
    actor_message_release(&message);
    printf("Forward solving process stopped!\n");

    // calc images
    char buffer[1024];
    for (linalgcl_size_t i = 0; i < solver->drive_count; i++) {
        // copy current phi to vector
        ert_solver_copy_from_column(solver, solver->applied_phi, solver->phi,
            i, queue0);

        // calc image
        ert_image_calc(image, solver->phi, queue0);
        clFinish(queue0);
        linalgcl_matrix_copy_to_host(image->image, queue0, CL_TRUE);
        linalgcl_matrix_save("output/image.txt", image->image);
        sprintf(buffer, "python src/script.py %d", i);
        system(buffer);
    }

    // voltage
    linalgcl_matrix_copy_to_host(solver->calculated_voltage, queue0, CL_TRUE);
    linalgcl_matrix_save("output/voltage.txt", solver->calculated_voltage);

    // cleanup
    ert_solver_release(&solver);
    ert_image_release(&image);
    clReleaseCommandQueue(queue0);
    clReleaseCommandQueue(queue1);
    clReleaseContext(context);

    return ACTOR_SUCCESS;
}

int main(int argc, char* argv[]) {
    // create node
    actor_node_t node = NULL;
    if (actor_node_create(&node, 0, 10) != ACTOR_SUCCESS) {
        return EXIT_FAILURE;
    }

    // start main process
    actor_spawn(node, NULL, ^actor_error_t(actor_process_t self) {
            // call main process
            actor_error_t error = main_process(self);

            // print result
            printf("main process died with result: %s!\n", actor_error_string(error));

            return error;
        });

    // wait for processes to complete
    while (actor_node_wait_for_processes(node, 10.0f) == ACTOR_ERROR_TIMEOUT) {

    }

    // release node
    actor_node_release(&node);

    return EXIT_SUCCESS;
};
