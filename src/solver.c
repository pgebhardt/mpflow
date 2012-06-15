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
#include <linalg/matrix.h>
#include <linalg/matrix_operations.h>
#include "basis.h"
#include "mesh.h"
#include "solver.h"

// create solver
linalg_error_t ert_solver_create(ert_solver_t* solverPointer, ert_mesh_t mesh) {
    // check input
    if ((solverPointer == NULL) || (mesh == NULL)) {
        return LINALG_ERROR;
    }

    // error
    linalg_error_t error = LINALG_SUCCESS;

    // init solver pointer to NULL
    *solverPointer = NULL;

    // create solver struct
    ert_solver_t solver = NULL;
    solver = malloc(sizeof(ert_solver_s));

    // check success
    if (solver == NULL) {
        return LINALG_ERROR;
    }

    // init struct
    solver->mesh = mesh;
    solver->A = NULL;
    solver->B = NULL;

    // create matrices
    error = linalg_matrix_create(&solver->A, solver->mesh->vertex_count,
        solver->mesh->vertex_count);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        ert_solver_release(&solver);

        return error;
    }

    // init A to 0.0
    for (linalg_size_t i = 0; i < solver->A->size_x; i++) {
        for (linalg_size_t j = 0; j < solver->A->size_y; j++) {
            linalg_matrix_set_element(solver->A, 0.0, i, j);
        }
    }

    // calc matrix elements for uniform sigma
    linalg_matrix_data_t x[3], y[3];
    linalg_matrix_data_t id[3];
    linalg_matrix_data_t sigma = 1.0;
    linalg_matrix_data_t space = 1.0;
    linalg_matrix_data_t element = 0.0;
    ert_basis_t basis[3];

    for (linalg_size_t k = 0; k < solver->mesh->element_count; k++) {
        // get vertices for element
        for (linalg_size_t i = 0; i < 3; i++) {
            linalg_matrix_get_element(solver->mesh->elements, &id[i], k, i);
            linalg_matrix_get_element(solver->mesh->vertices, &x[i], (linalg_size_t)id[i], 0);
            linalg_matrix_get_element(solver->mesh->vertices, &y[i], (linalg_size_t)id[i], 1);
        }

        // calc corresponding basis functions
        ert_basis_create(&basis[0], x[0], y[0], x[1], y[1], x[2], y[2]);
        ert_basis_create(&basis[1], x[1], y[1], x[2], y[2], x[0], y[0]);
        ert_basis_create(&basis[2], x[2], y[2], x[0], y[0], x[1], y[1]);

        // calc matrix elements
        // TODO: calc space of triangle
        element = 0.0;
        for (linalg_size_t i = 0; i < 3; i++) {
            for (linalg_size_t j = 0; j < 3; j++) {
                // get current value
                linalg_matrix_get_element(solver->A, &element,
                    (linalg_size_t)id[i], (linalg_size_t)id[j]);

                linalg_matrix_set_element(solver->A, sigma * space *
                    (basis[i]->gradient[0] * basis[j]->gradient[0] +
                     basis[i]->gradient[1] * basis[j]->gradient[1]) + element,
                     (linalg_size_t)id[i], (linalg_size_t)id[j]);
            }
        }

        // cleanup
        ert_basis_release(&basis[0]);
        ert_basis_release(&basis[1]);
        ert_basis_release(&basis[2]);
    }

    // set solver pointer
    *solverPointer = solver;

    return LINALG_SUCCESS;
}

// release solver
linalg_error_t ert_solver_release(ert_solver_t* solverPointer) {
    // check input
    if ((solverPointer == NULL) || (*solverPointer == NULL)) {
        return LINALG_ERROR;
    }

    // get solver
    ert_solver_t solver = *solverPointer;

    // cleanup
    ert_mesh_release(&solver->mesh);
    linalg_matrix_release(&solver->A);
    linalg_matrix_release(&solver->B);

    // free struct
    free(solver);

    // set solver pointer to NULL
    *solverPointer = NULL;

    return LINALG_SUCCESS;
}

