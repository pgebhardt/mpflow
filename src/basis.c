// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fasteit.h"

// create basis
linalgcuError_t fasteit_basis_create(fasteitBasis_t* basisPointer,
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
    fasteitBasis_t self = malloc(sizeof(fasteitBasis_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->points[0][0] = Ax;
    self->points[0][1] = Ay;
    self->points[1][0] = Bx;
    self->points[1][1] = By;
    self->points[2][0] = Cx;
    self->points[2][1] = Cy;
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
linalgcuError_t fasteit_basis_release(fasteitBasis_t* basisPointer) {
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
linalgcuError_t fasteit_basis_function(fasteitBasis_t self,
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

// integrate with basis
linalgcuMatrixData_t fasteit_basis_integrate_with_basis(fasteitBasis_t self, fasteitBasis_t other) {
    // shorten variables
    linalgcuMatrixData_t x1 = self->points[0][0];
    linalgcuMatrixData_t y1 = self->points[0][1];
    linalgcuMatrixData_t x2 = self->points[1][0];
    linalgcuMatrixData_t y2 = self->points[1][1];
    linalgcuMatrixData_t x3 = self->points[2][0];
    linalgcuMatrixData_t y3 = self->points[2][1];

    linalgcuMatrixData_t ai = self->coefficients[0];
    linalgcuMatrixData_t bi = self->coefficients[1];
    linalgcuMatrixData_t ci = self->coefficients[2];
    linalgcuMatrixData_t aj = other->coefficients[0];
    linalgcuMatrixData_t bj = other->coefficients[1];
    linalgcuMatrixData_t cj = other->coefficients[2];

    // calc area
    linalgcuMatrixData_t area = 0.5 * fabs((x2 - x1) * (y3 - y1) -
        (x3 - x1) * (y2 - y1));

    // calc integral
    linalgcuMatrixData_t integral = 2.0f * area *
        (ai * (0.5f * aj + (1.0f / 6.0f) * bj * (x1 + x2 + x3) +
        (1.0f / 6.0f) * cj * (y1 + y2 + y3)) +
        bi * ((1.0f/ 6.0f) * aj * (x1 + x2 + x3) +
        (1.0f / 12.0f) * bj * (x1 * x1 + x1 * x2 + x1 * x3 + x2 * x2 + x2 * x3 + x3 * x3) +
        (1.0f/ 24.0f) * cj * (2.0f * x1 * y1 + x1 * y2 + x1 * y3 + x2 * y1 +
        2.0f * x2 * y2 + x2 * y3 + x3 * y1 + x3 * y2 + 2.0f * x3 * y3)) +
        ci * ((1.0f / 6.0f) * aj * (y1 + y2 + y3) +
        (1.0f / 12.0f) * cj * (y1 * y1 + y1 * y2 + y1 * y3 + y2 * y2 + y2 * y3 + y3 * y3) +
        (1.0f / 24.0f) * bj * (2.0f * x1 * y1 + x1 * y2 + x1 * y3 + x2 * y1 +
        2.0f * x2 * y2 + x2 * y3 + x3 * y1 + x3 * y2 + 2.0f * x3 * y3)));

    return integral;
}

