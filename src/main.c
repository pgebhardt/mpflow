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
#include <actor/actor.h>
#include <linalg/matrix.h>
#include "mesh.h"
#include "basis.h"
#include "solver.h"

void print_matrix(linalg_matrix_t matrix) {
    if (matrix == NULL) {
        return;
    }

    // value memory
    linalg_matrix_data_t value = 0.0;

    for (linalg_size_t i = 0; i < matrix->size_x; i++) {
        for (linalg_size_t j = 0; j < matrix->size_y; j++) {
            // get value
            linalg_matrix_get_element(matrix, &value, i, j);

            printf("%f, ", value);
        }
        printf("\n");
    }
}

static actor_process_function_t main_process = ^(actor_process_t self) {
    // error
    linalg_error_t error = LINALG_SUCCESS;

    // create mesh
    ert_mesh_t mesh = NULL;
    error = ert_mesh_create(&mesh, 1.0, 1.0 / 16.0);

    // check success
    if (error != LINALG_SUCCESS) {
        return ACTOR_ERROR;
    }

    // create solver
    ert_solver_t solver = NULL;
    error = ert_solver_create(&solver, mesh);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        ert_mesh_release(&mesh);

        return ACTOR_ERROR;
    }

    // save vertices
    linalg_matrix_save("vertices.txt", mesh->vertices);

    // save elements
    linalg_matrix_save("elements.txt", mesh->elements);

    // show grid
    system("python src/script.py");

    /*printf("vertex count: %d\n", mesh->vertex_count);
    printf("element count: %d\n", mesh->element_count);
    printf("A: \n");
    print_matrix(solver->A);*/

    // cleanup
    ert_solver_release(&solver);

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
