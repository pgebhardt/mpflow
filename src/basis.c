// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create basis
linalgcuError_t fastect_basis_create(fastectBasis_t* basisPointer,
    linalgcuMatrixData_t Ax, linalgcuMatrixData_t Ay,
    linalgcuMatrixData_t Bx, linalgcuMatrixData_t By,
    linalgcuMatrixData_t Cx, linalgcuMatrixData_t Cy) {
    // check input
    if (basisPointer == NULL) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // init basis pointer
    *basisPointer = NULL;

    // create basis struct
    fastectBasis_t self = malloc(sizeof(fastectBasis_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->coefficients[0] = 0.0;
    self->coefficients[1] = 0.0;
    self->coefficients[2] = 0.0;
    self->gradient[0] = 0.0;
    self->gradient[1] = 0.0;

    // calc coefficients (A * c = b)
    linalgcuMatrixData_t Ainv[3][3];
    linalgcuMatrixData_t B[3] = {1.0, 0.0, 0.0};

    // invert matrix A directly
    linalgcuMatrixData_t a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = Ax;
    c = Ay;
    d = 1.0;
    e = Bx;
    f = By;
    g = 1.0;
    h = Cx;
    i = Cy;
    linalgcuMatrixData_t det = a * (e * i - f * h) - b * (i * d - f * g) + c * (d * h - e * g);

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
    self->coefficients[0] = Ainv[0][0] * B[0] + Ainv[0][1] * B[1] + Ainv[0][2] * B[2];
    self->coefficients[1] = Ainv[1][0] * B[0] + Ainv[1][1] * B[1] + Ainv[1][2] * B[2];
    self->coefficients[2] = Ainv[2][0] * B[0] + Ainv[2][1] * B[1] + Ainv[2][2] * B[2];

    // save gradient
    self->gradient[0] = self->coefficients[1];
    self->gradient[1] = self->coefficients[2];

    // set basis pointer
    *basisPointer = self;

    return LINALGCU_SUCCESS;
}

// release basis
linalgcuError_t fastect_basis_release(fastectBasis_t* basisPointer) {
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
linalgcuError_t fastect_basis_function(fastectBasis_t self,
    linalgcuMatrixData_t* resultPointer, linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    // check input
    if ((self == NULL) || (resultPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // calc result
    *resultPointer = self->coefficients[0] + self->coefficients[1] * x +
        self->coefficients[2] * y;

    return LINALGCU_SUCCESS;
}
