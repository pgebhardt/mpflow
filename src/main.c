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

int main(int argc, char* argv[]) {
    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;
    cl_int cl_error = CL_SUCCESS;

    // Connect to a compute device
    cl_device_id device_id;
    cl_error = clGetDeviceIDs(NULL, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);

    // check success
    if (cl_error != CL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // Create a compute context 
    cl_context context = clCreateContext(0, 1, &device_id, NULL, NULL, &cl_error);

    // check success
    if (cl_error != CL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // Create a command commands
    cl_command_queue queue = clCreateCommandQueue(context, device_id, 0, &cl_error);

    if (cl_error != CL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create matrix program
    linalgcl_matrix_program_t program = NULL;
    error = linalgcl_matrix_create_programm(&program, context, device_id,
        "/usr/local/include/linalgcl/matrix.cl");

    // check success
    if (error != LINALGCL_SUCCESS) {
        printf("Matrix programm laden ging nicht!\n");
        return EXIT_FAILURE;
    }

    // create mesh
    ert_mesh_t mesh;
    error = ert_mesh_create(&mesh, 0.045, 0.045 / 16.0, context);

    // check success
    if (error != LINALGCL_SUCCESS) {
        printf("Mesh erzeugen ging nicht!\n");
        return EXIT_FAILURE;
    }

    // copy matrices to device
    linalgcl_matrix_copy_to_device(mesh->vertices, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(mesh->elements, queue, CL_TRUE);

    // create electrodes
    ert_electrodes_t electrodes;
    error = ert_electrodes_create(&electrodes, 36, 0.005, mesh);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create solver
    ert_solver_t solver;
    error = ert_solver_create(&solver, mesh, electrodes, 9, 18,
        program, context, device_id, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        printf("Kann keinen Solver erstellen!\n");
        return EXIT_FAILURE;
    }

    // Create image
    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh, context, device_id);
    linalgcl_matrix_copy_to_device(image->elements, queue, CL_FALSE);
    linalgcl_matrix_copy_to_device(image->image, queue, CL_TRUE);

    // set sigma
    for (linalgcl_size_t i = 0; i < mesh->element_count; i++) {
        if (i < mesh->element_count / 2) {
            linalgcl_matrix_set_element(solver->sigma, 15.0f * 1E-3, i, 0);
        }
        else {
            linalgcl_matrix_set_element(solver->sigma, 10.0f * 1E-3, i, 0);
        }
    }
    linalgcl_matrix_copy_to_device(solver->sigma, queue, CL_TRUE);
    ert_grid_update_system_matrix(solver->grid, queue);

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    clFinish(queue);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // solve
    ert_solver_forward_solve(solver, program, queue);

    // get end time
    clFinish(queue);
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;
    printf("Solving time: %f ms\n", (end - start) * 1E3);

    // calc image
    ert_image_calc(image, solver->sigma, queue);
    clFinish(queue);
    linalgcl_matrix_copy_to_host(image->image, queue, CL_TRUE);
    linalgcl_matrix_save("image.txt", image->image);
    system("python src/script.py");

    // voltage
    linalgcl_matrix_t voltage, potential;
    linalgcl_matrix_create(&voltage, context, solver->measurment_pattern->size_x, 18);
    linalgcl_matrix_create(&potential, context, electrodes->count, 18);

    linalgcl_matrix_multiply(potential, solver->voltage_calculation, solver->applied_phi,
        program, queue);
    linalgcl_matrix_multiply(voltage, solver->measurment_pattern, potential, program, queue);
    clFinish(queue);

    linalgcl_matrix_copy_to_host(voltage, queue, CL_TRUE);
    linalgcl_matrix_save("voltage.txt", voltage);
    linalgcl_matrix_release(&voltage);
    linalgcl_matrix_release(&potential);

    linalgcl_matrix_copy_to_host(solver->jacobian, queue, CL_TRUE);
    linalgcl_matrix_save("jacobian.txt", solver->jacobian);

    // cleanup
    ert_solver_release(&solver);
    ert_image_release(&image);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
};
