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
    // TODO: Leider werden nicht beide GPUs verwendet
    cl_command_queue queue0 = clCreateCommandQueue(context, device_id[0], 0, &cl_error);
    cl_command_queue queue1 = clCreateCommandQueue(context, device_id[1], 0, &cl_error);

    if (cl_error != CL_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create matrix program
    linalgcl_matrix_program_t program0, program1;
    error  = linalgcl_matrix_create_programm(&program0, context, device_id[0],
        "/usr/local/include/linalgcl/matrix.cl");
    error |= linalgcl_matrix_create_programm(&program1, context, device_id[1],
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
        program0, context, device_id[0], queue0);

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
        linalgcl_matrix_set_element(solver->sigma, 1.0f, i, 0);
    }
    linalgcl_matrix_copy_to_device(solver->sigma, queue0, CL_TRUE);
    ert_grid_update_system_matrix(solver->grid, queue0);
    clFinish(queue0);

    // start forward solving process
    actor_process_id_t forward = ACTOR_INVALID_ID;
    actor_spawn(main->node, &forward, ^actor_error_t(actor_process_t self) {
        // link to main process
        actor_process_link(self, main->nid, main->pid);

        return ert_solver_forward(self, solver, program0, context, queue0);
    });
    printf("Forward solving process started!\n");

    actor_process_sleep(main, 0.5);

    // start inverse solving process
    actor_process_id_t inverse = ACTOR_INVALID_ID;
    /*actor_spawn(main->node, &inverse, ^actor_error_t(actor_process_t self) {
        // link to main process
        actor_process_link(self, main->nid, main->pid);

        return ert_solver_inverse(self, solver, program1, context, queue1);
    });*/
    printf("Inverse solving process started!\n");

    // sleep a bit
    actor_process_sleep(main, 10.0);

    // stop forward process
    actor_send(main, main->nid, forward, ACTOR_TYPE_ERROR_MESSAGE,
        &forward, sizeof(actor_process_id_t));
    printf("Send stop signal to forward solving process!\n");

    // wait for forward process to stop
    actor_message_t message = NULL;
    actor_receive(main, &message, 2.0);
    actor_message_release(&message);
    printf("Forward solving process stopped!\n");

    // stop inverse process
    actor_send(main, main->nid, inverse, ACTOR_TYPE_ERROR_MESSAGE,
        &forward, sizeof(actor_process_id_t));
    printf("Send stop signal to inverse solving process!\n");

    // wait for forward process to stop
    actor_receive(main, &message, 2.0);
    actor_message_release(&message);
    printf("Inverse solving process stopped!\n");

    /*// create buffer
    linalgcl_matrix_t phi;
    linalgcl_matrix_create(&phi, context, solver->applied_phi->size_x, 1);

    // calc images
    char buffer[1024];
    for (linalgcl_size_t i = 0; i < solver->drive_count; i++) {
        // copy current phi to vector
        ert_solver_copy_from_column(solver->program0, solver->applied_phi, phi,
            i, queue0);

        // calc image
        ert_image_calc_phi(image, phi, queue0);
        clFinish(queue0);
        linalgcl_matrix_copy_to_host(image->image, queue0, CL_TRUE);
        linalgcl_matrix_save("output/image.txt", image->image);
        sprintf(buffer, "python src/script.py %d", i);
        system(buffer);
    }
    linalgcl_matrix_release(&phi);

    // calc sigma image
    ert_image_calc_sigma(image, solver->sigma, queue0);
    clFinish(queue0);
    linalgcl_matrix_copy_to_host(image->image, queue0, CL_TRUE);
    linalgcl_matrix_save("output/image.txt", image->image);
    sprintf(buffer, "python src/script.py %d", solver->drive_count);
    system(buffer);

    // save some data
    linalgcl_matrix_copy_to_host(solver->calculated_voltage, queue0, CL_TRUE);
    linalgcl_matrix_save("output/voltage.txt", solver->calculated_voltage);
    linalgcl_matrix_copy_to_host(solver->gradient, queue0, CL_TRUE);
    linalgcl_matrix_save("output/gradient.txt", solver->gradient);
    linalgcl_matrix_copy_to_host(solver->jacobian, queue0, CL_TRUE);
    linalgcl_matrix_save("output/jacobian.txt", solver->jacobian);
    linalgcl_matrix_copy_to_host(solver->sigma, queue0, CL_TRUE);
    linalgcl_matrix_save("output/sigma.txt", solver->sigma);*/

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
