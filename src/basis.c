// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create basis
linalgcu_error_t fastect_basis_create(fastect_basis_t* basisPointer,
    linalgcu_matrix_data_t Ax, linalgcu_matrix_data_t Ay,
    linalgcu_matrix_data_t Bx, linalgcu_matrix_data_t By,
    linalgcu_matrix_data_t Cx, linalgcu_matrix_data_t Cy) {
    // check input
    if (basisPointer == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // init basis pointer
    *basisPointer = NULL;

    // create basis struct
    fastect_basis_t basis = malloc(sizeof(fastect_basis_s));

    // check success
    if (basis == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    basis->coefficients[0] = 0.0;
    basis->coefficients[1] = 0.0;
    basis->coefficients[2] = 0.0;
    basis->gradient[0] = 0.0;
    basis->gradient[1] = 0.0;

    // calc coefficients (A * c = b)
    linalgcu_matrix_data_t Ainv[3][3];
    linalgcu_matrix_data_t B[3] = {1.0, 0.0, 0.0};

    // invfastect matrix A directly
    linalgcu_matrix_data_t a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = Ax;
    c = Ay;
    d = 1.0;
    e = Bx;
    f = By;
    g = 1.0;
    h = Cx;
    i = Cy;
    linalgcu_matrix_data_t det = a * (e * i - f * h) - b * (i * d - f * g) + c * (d * h - e * g);

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

    return LINALGCU_SUCCESS;
}

// release basis
linalgcu_error_t fastect_basis_release(fastect_basis_t* basisPointer) {
    // check input
    if ((basisPointer == NULL) || (*basisPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // free struct
    free(*basisPointer);

    // set basis pointer to NULL
    *basisPointer = NULL;

    return LINALGCU_SUCCESS;
}

// evaluate basis function
linalgcu_error_t fastect_basis_function(fastect_basis_t basis, linalgcu_matrix_data_t* resultPointer,
    linalgcu_matrix_data_t x, linalgcu_matrix_data_t y) {
    // check input
    if ((basis == NULL) || (resultPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // calc result
    *resultPointer = basis->coefficients[0] + basis->coefficients[1] * x +
        basis->coefficients[2] * y;

    return LINALGCU_SUCCESS;
}
