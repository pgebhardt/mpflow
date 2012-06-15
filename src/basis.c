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
    linalg_matrix_t Ainv, B, C;
    error = linalg_matrix_create(&Ainv, 3, 3);
    error += linalg_matrix_create(&C, 3, 1);
    error += linalg_matrix_create(&B, 3, 1);

    // check success
    if (error != LINALG_SUCCESS) {
        // cleanup
        linalg_matrix_release(&Ainv);
        linalg_matrix_release(&C);
        linalg_matrix_release(&B);

        return LINALG_ERROR;
    }

    // fill result vector b
    linalg_matrix_set_element(B, 1.0, 0, 0);
    linalg_matrix_set_element(B, 0.0, 1, 0);
    linalg_matrix_set_element(B, 0.0, 2, 0);

    // invert matrix A directly
    linalg_matrix_data_t a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = Ax;
    c = Ay;
    d = 1.0;
    e = Bx;
    f = By;
    g = 1.0;
    h = Cx;
    i = Cy;
    linalg_matrix_data_t det = a * (e * i - f * h) - b * (i * d - f * g) + c * (d * h - e * g);

    linalg_matrix_set_element(Ainv, (e * i - f * h) / det, 0, 0);
    linalg_matrix_set_element(Ainv, (c * h - b * i) / det, 0, 1);
    linalg_matrix_set_element(Ainv, (b * f - c * e) / det, 0, 2);
    linalg_matrix_set_element(Ainv, (f * g - d * i) / det, 1, 0);
    linalg_matrix_set_element(Ainv, (a * i - c * g) / det, 1, 1);
    linalg_matrix_set_element(Ainv, (c * d - a * f) / det, 1, 2);
    linalg_matrix_set_element(Ainv, (d * h - e * g) / det, 2, 0);
    linalg_matrix_set_element(Ainv, (g * b - a * h) / det, 2, 1);
    linalg_matrix_set_element(Ainv, (a * e - b * d) / det, 2, 2);

    // calc coefficients
    linalg_matrix_multiply(&C, Ainv, B);

    printf("basis zu: A = (%f, %f), B = (%f, %f) C = (%f, %f): %f, %f, %f\n",
        Ax, Ay, Bx, By, Cx, Cy, C->data[0], C->data[1], C->data[2]);

    // save coefficients
    linalg_matrix_get_element(C, &basis->coefficients[0], 0, 0);
    linalg_matrix_get_element(C, &basis->coefficients[1], 1, 0);
    linalg_matrix_get_element(C, &basis->coefficients[2], 2, 0);

    // save gradient
    basis->gradient[0] = basis->coefficients[1];
    basis->gradient[1] = basis->coefficients[2];

    // cleanup
    linalg_matrix_release(&Ainv);
    linalg_matrix_release(&B);
    linalg_matrix_release(&C);

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

// evaluate basis function
linalg_error_t ert_basis_function(ert_basis_t basis, linalg_matrix_data_t* resultPointer,
    linalg_matrix_data_t x, linalg_matrix_data_t y) {
    // check input
    if ((basis == NULL) || (resultPointer == NULL)) {
        return LINALG_ERROR;
    }

    // calc result
    *resultPointer = basis->coefficients[0] + basis->coefficients[1] * x +
        basis->coefficients[2] * y;

    return LINALG_SUCCESS;
}
