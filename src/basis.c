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
#include <math.h>
#include <linalg/matrix.h>
#include <linalg/matrix_operations.h>
#include "basis.h"

// create basis
linalg_error_t ert_basis_create(ert_basis_t* basisPointer,
    linalg_matrix_data_t Ax, linalg_matrix_data_t Ay,
    linalg_matrix_data_t Bx, linalg_matrix_data_t By,
    linalg_matrix_data_t Cx, linalg_matrix_data_t Cy) {
    // check input
    if (basisPointer == NULL) {
        return LINALG_ERROR;
    }

    // error
    linalg_error_t error = LINALG_SUCCESS;

    // init basis pointer
    *basisPointer = NULL;

    // create basis struct
    ert_basis_t basis = malloc(sizeof(ert_basis_s));

    // check success
    if (basis == NULL) {
        return LINALG_ERROR;
    }

    // init struct
    basis->coefficients[0] = 0.0;
    basis->coefficients[1] = 0.0;
    basis->coefficients[2] = 0.0;
    basis->gradient[0] = 0.0;
    basis->gradient[1] = 0.0;

    // calc coefficients (A * c = b)
    linalg_matrix_t A, b, c;
    error = linalg_matrix_create(&A, 3, 3);
    error += linalg_matrix_create(&c, 3, 1);
    error += linalg_matrix_create(&b, 3, 1);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        linalg_matrix_release(&A);
        linalg_matrix_release(&c);
        linalg_matrix_release(&b);

        return LINALG_ERROR;
    }

    // fill Matrix A
    linalg_matrix_set_element(A, 1.0, 0, 0);
    linalg_matrix_set_element(A, 1.0, 1, 0);
    linalg_matrix_set_element(A, 1.0, 2, 0);
    linalg_matrix_set_element(A, Ax, 0, 1);
    linalg_matrix_set_element(A, Bx, 1, 1);
    linalg_matrix_set_element(A, Cx, 2, 1);
    linalg_matrix_set_element(A, Ay, 0, 2);
    linalg_matrix_set_element(A, By, 1, 2);
    linalg_matrix_set_element(A, Cy, 2, 2);

    // fill result vector b
    linalg_matrix_set_element(b, 1.0, 0, 0);
    linalg_matrix_set_element(A, 0.0, 1, 0);
    linalg_matrix_set_element(A, 0.0, 2, 0);

    // calc coefficients
    linalg_matrix_t temp;
    linalg_matrix_inverse(&temp, A);
    linalg_matrix_multiply(&c, temp, b);

    // save coefficients
    linalg_matrix_get_element(c, &basis->coefficients[0], 0, 0);
    linalg_matrix_get_element(c, &basis->coefficients[1], 1, 0);
    linalg_matrix_get_element(c, &basis->coefficients[2], 2, 0);

    // save gradient
    basis->gradient[0] = basis->coefficients[1];
    basis->gradient[1] = basis->coefficients[2];

    // cleanup
    linalg_matrix_release(&temp);
    linalg_matrix_release(&A);
    linalg_matrix_release(&b);
    linalg_matrix_release(&c);

    // set basis pointer
    *basisPointer = basis;

    return LINALG_SUCCESS;
}

// release basis
linalg_error_t ert_basis_release(ert_basis_t* basisPointer) {
    // check input
    if ((basisPointer == NULL) || (*basisPointer == NULL)) {
        return LINALG_ERROR;
    }

    // free struct
    free(*basisPointer);

    // set basis pointer to NULL
    *basisPointer = NULL;

    return LINALG_SUCCESS;
}
