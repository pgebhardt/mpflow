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
    error = ert_mesh_create(&mesh, 1.0, 1.0 / 18.0, context);
    printf("Generated mesh!\n");

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
    error = ert_electrodes_create(&electrodes, 36, 0.08, mesh);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create grid
    ert_grid_t grid = NULL;
    error  = ert_grid_create(&grid, program, mesh, context, device_id, queue);
    error |= ert_grid_init_exitation_matrix(grid, electrodes, context, queue);

    // check success
    if (error != LINALGCL_SUCCESS) {
        return EXIT_FAILURE;
    }

    // create solver
    ert_gradient_solver_t solver = NULL;
    error |= ert_gradient_solver_create(&solver, grid->system_matrix,
        mesh->vertex_count, program, context, device_id, queue);

    printf("Created solver and grids!\n");

    // check success
    if (error != LINALGCL_SUCCESS) {
        printf("Solver erzeugen ging nicht!\n");
        return EXIT_FAILURE;
    }

    // create image
    ert_image_t image;
    ert_image_create(&image, 1000, 1000, mesh, context, device_id);
    linalgcl_matrix_copy_to_device(image->elements, queue, CL_TRUE);
    linalgcl_matrix_copy_to_device(image->image, queue, CL_TRUE);

    // x vector
    linalgcl_matrix_t x;
    linalgcl_matrix_create(&x, context, mesh->vertex_count, 1);
    linalgcl_matrix_copy_to_device(x, queue, CL_TRUE);

    // right hand side
    linalgcl_matrix_t j, f;
    linalgcl_matrix_create(&j, context, electrodes->count, 1);
    linalgcl_matrix_create(&f, context, mesh->vertex_count, 1);

    // set j
    linalgcl_matrix_set_element(j, 1.0f, 0, 0);
    linalgcl_matrix_set_element(j, -1.0f, 2, 0);
    linalgcl_matrix_copy_to_device(j, queue, CL_TRUE);

    // calc f matrix
    // f = B * j
    linalgcl_matrix_multiply(f, grid->exitation_matrix, j, program, queue);
    printf("Calced start vector and right side!\n");

    // get start time
    struct timeval tv;
    gettimeofday(&tv, NULL);
    clFinish(queue);
    double start = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;

    // solve
    ert_gradient_solver_solve_singular(solver, x, f, 1E-5, program, queue);

    // get end time
    clFinish(queue);
    gettimeofday(&tv, NULL);
    double end = (double)tv.tv_sec + (double)tv.tv_usec / 1E6;
    printf("Solving time: %f ms\n", (end - start) * 1E3);

    ert_image_calc(image, x, queue);
    clFinish(queue);
    linalgcl_matrix_copy_to_host(image->image, queue, CL_TRUE);
    linalgcl_matrix_save("image.txt", image->image);
    system("python src/script.py");
    system("rm image.txt");

    // cleanup
    linalgcl_matrix_release(&x);
    linalgcl_matrix_release(&f);
    linalgcl_matrix_release(&j);
    ert_gradient_solver_release(&solver);
    ert_grid_release(&grid);
    ert_electrodes_release(&electrodes);
    ert_image_release(&image);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return EXIT_SUCCESS;
};
