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
#include "mesh.h"
#include "basis.h"
#include "image.h"
#include "electrodes.h"
#include "grid.h"
#include "conjugate.h"
#include "forward.h"

int main(int argc, char* argv[]) {
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
        return EXIT_FAILURE;
    }

    // Create a compute context 
    cl_context context = clCreateContext(0, 2, device_id, NULL, NULL, &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // Create a command commands
    cl_command_queue queue0, queue1;
    queue0 = clCreateCommandQueue(context, device_id[0], 0, &cl_error);
    queue1 = clCreateCommandQueue(context, device_id[1], 0, &cl_error);

    if (cl_error != CL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create matrix program
    linalgcl_matrix_program_t program0, program1;
    error  = linalgcl_matrix_create_programm(&program0, context, device_id[0],
        "/usr/local/include/linalgcl/matrix.cl");
    error |= linalgcl_matrix_create_programm(&program1, context, device_id[1],
        "/usr/local/include/linalgcl/matrix.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create mesh
    ert_mesh_t mesh;
    error  = ert_mesh_create(&mesh, 0.045, 0.045 / 16.0, context);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(mesh->vertices, queue0, CL_FALSE);
    linalgcl_matrix_copy_to_device(mesh->elements, queue0, CL_TRUE);

    // create electrodes
    ert_electrodes_t electrodes;
    error  = ert_electrodes_create(&electrodes, 36, 0.005, mesh);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // load drive pattern
    linalgcl_matrix_t drive_pattern;
    linalgcl_matrix_load(&drive_pattern, context, "input/drive_pattern.txt");
    linalgcl_matrix_copy_to_device(drive_pattern, queue0, CL_TRUE);

    // create solver
    ert_forward_solver_t solver;
    error  = ert_forward_solver_create(&solver, mesh, electrodes, 18, drive_pattern,
        program0, context, device_id[0], queue0);

    // check success
    if (error != LINALGCL_SUCCESS) {
        printf("Solver erstellen ging nicht!\n");
        return EXIT_FAILURE;
    }

    // Create image
    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh, context, device_id[0]);
    linalgcl_matrix_copy_to_device(image->elements, queue1, CL_FALSE);
    linalgcl_matrix_copy_to_device(image->image, queue1, CL_TRUE);

    // get start time
    struct timeval tv;
    clFinish(queue0);
    clFinish(queue1);
    gettimeofday(&tv, NULL);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    for (linalgcl_size_t i = 0; i < 100; i++) {
        ert_forward_solver_solve(solver, program0, queue0);
    }

    // get end time
    clFinish(queue0);
    clFinish(queue1);
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    printf("Frames per second: %f\n", 100.0 / (end - start));

    /*// create buffer
    linalgcl_matrix_t phi;
    linalgcl_matrix_create(&phi, context, solver->phi->size_x, 1);

    // calc images
    char buffer[1024];
    for (linalgcl_size_t i = 0; i < solver->count; i++) {
        // copy current phi to vector
        ert_forward_solver_copy_from_column(solver->program, solver->phi, phi,
            i, queue0);

        // calc image
        ert_image_calc_phi(image, phi, queue0);
        clFinish(queue0);
        linalgcl_matrix_copy_to_host(image->image, queue0, CL_TRUE);
        linalgcl_matrix_save("output/image.txt", image->image);
        sprintf(buffer, "python src/script.py %d", i);
        system(buffer);
    }
    linalgcl_matrix_release(&phi);*/

    // cleanup
    linalgcl_matrix_program_release(&program0);
    linalgcl_matrix_program_release(&program1);
    ert_forward_solver_release(&solver);
    ert_image_release(&image);
    clReleaseCommandQueue(queue0);
    clReleaseCommandQueue(queue1);
    clReleaseContext(context);

    return EXIT_SUCCESS;
};
