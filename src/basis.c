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

#include <linalgcl/linalgcl.h>
#include "basis.h"

// create basis
linalgcl_error_t ert_basis_create(ert_basis_t* basisPointer,
    linalgcl_matrix_data_t Ax, linalgcl_matrix_data_t Ay,
    linalgcl_matrix_data_t Bx, linalgcl_matrix_data_t By,
    linalgcl_matrix_data_t Cx, linalgcl_matrix_data_t Cy) {
    // check input
    if (basisPointer == NULL) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // init basis pointer
    *basisPointer = NULL;

    // create basis struct
    ert_basis_t basis = malloc(sizeof(ert_basis_s));

    // check success
    if (basis == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    basis->coefficients[0] = 0.0;
    basis->coefficients[1] = 0.0;
    basis->coefficients[2] = 0.0;
    basis->gradient[0] = 0.0;
    basis->gradient[1] = 0.0;

    // calc coefficients (A * c = b)
    linalgcl_matrix_data_t Ainv[3][3];
    linalgcl_matrix_data_t B[3] = {1.0, 0.0, 0.0};

    // invert matrix A directly
    linalgcl_matrix_data_t a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = Ax;
    c = Ay;
    d = 1.0;
    e = Bx;
    f = By;
    g = 1.0;
    h = Cx;
    i = Cy;
    linalgcl_matrix_data_t det = a * (e * i - f * h) - b * (i * d - f * g) + c * (d * h - e * g);

    Ainv[0][0] = (e * i - f * h) / det;
    Ainv[0][1] = (c * h - b * i) / det;
    Ainv[0][2] = (b * f - c * e) / det;
    Ainv[1][0] = (f * g - d * i) / det;
    Ainv[1][1] = (a * i - c * g) / det;
    Ainv[1][2] = (c * d - a * f) / det;
    Ainv[2][0] = (d * h - e * g) / det;
    Ainv[2][1] = (g * b - a * h) / det;
    Ainv[2][2] = (a * e - b * d) / det;

    // calc coefficients
    basis->coefficients[0] = Ainv[0][0] * B[0] + Ainv[0][1] * B[1] + Ainv[0][2] * B[2];
    basis->coefficients[1] = Ainv[1][0] * B[0] + Ainv[1][1] * B[1] + Ainv[1][2] * B[2];
    basis->coefficients[2] = Ainv[2][0] * B[0] + Ainv[2][1] * B[1] + Ainv[2][2] * B[2];

    // save gradient
    basis->gradient[0] = basis->coefficients[1];
    basis->gradient[1] = basis->coefficients[2];

    // set basis pointer
    *basisPointer = basis;

    return LINALGCL_SUCCESS;
}

// release basis
linalgcl_error_t ert_basis_release(ert_basis_t* basisPointer) {
    // check input
    if ((basisPointer == NULL) || (*basisPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // free struct
    free(*basisPointer);

    // set basis pointer to NULL
    *basisPointer = NULL;

    return LINALGCL_SUCCESS;
}

// evaluate basis function
linalgcl_error_t ert_basis_function(ert_basis_t basis, linalgcl_matrix_data_t* resultPointer,
    linalgcl_matrix_data_t x, linalgcl_matrix_data_t y) {
    // check input
    if ((basis == NULL) || (resultPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // calc result
    *resultPointer = basis->coefficients[0] + basis->coefficients[1] * x +
        basis->coefficients[2] * y;

    return LINALGCL_SUCCESS;
}
