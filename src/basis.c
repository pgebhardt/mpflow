// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fasteit.h"

// create basis
linalgcuError_t fasteit_basis_create(fasteitBasis_t* basisPointer,
    linalgcuMatrixData_t* x, linalgcuMatrixData_t* y) {
    // check input
    if ((basisPointer == NULL) || (x == NULL) || (y == NULL)) {
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
    for (linalgcuSize_t i = 0; i < FASTEIT_NODES_PER_ELEMENT; i++) {
        self->points[i][0] = x[i];
        self->points[i][1] = y[i];
        self->coefficients[i] = 0.0;
    }

    // calc coefficients (A * c = b)
    linalgcuMatrixData_t Ainv[3][3];
    linalgcuMatrixData_t B[3] = {1.0, 0.0, 0.0};

    // invert matrix A directly
    linalgcuMatrixData_t a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = self->points[0][0];
    c = self->points[0][1];
    d = 1.0;
    e = self->points[1][0];
    f = self->points[1][1];
    g = 1.0;
    h = self->points[2][0];
    i = self->points[2][1];
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
linalgcuMatrixData_t fasteit_basis_function(fasteitBasis_t self,
    linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    // check input
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // calc result
    return self->coefficients[0] + self->coefficients[1] * x +
        self->coefficients[2] * y;
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

// integrate gradient with basis
linalgcuMatrixData_t fasteit_basis_integrate_gradient_with_basis(fasteitBasis_t self,
    fasteitBasis_t other) {
    // calc area
    linalgcuMatrixData_t area = 0.5 * fabs((self->points[1][0] - self->points[0][0]) *
        (self->points[2][1] - self->points[0][1]) -
        (self->points[2][0] - self->points[0][0]) *
        (self->points[1][1] - self->points[0][1]));

    // calc integral
    return area * (self->coefficients[1] * other->coefficients[1] +
        self->coefficients[2] * other->coefficients[2]);
}

// angle calculation for parametrisation
linalgcuMatrixData_t fasteit_basis_angle(linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    if (x > 0.0f) {
        return atan(y / x);
    }
    else if ((x < 0.0f) && (y >= 0.0f)) {
        return atan(y / x) + M_PI;
    }
    else if ((x < 0.0f) && (y < 0.0f)) {
        return atan(y / x) - M_PI;
    }
    else if ((x == 0.0f) && (y > 0.0f)) {
        return M_PI / 2.0f;
    }
    else if ((x == 0.0f) && (y < 0.0f)) {
        return - M_PI / 2.0f;
    }
    else {
        return 0.0f;
    }
}

// integrate edge
linalgcuMatrixData_t fasteit_basis_integrate_edge(linalgcuMatrixData_t* x, linalgcuMatrixData_t* y,
    linalgcuMatrixData_t* start, linalgcuMatrixData_t* end) {
    // check input
    if ((x == NULL) || (y == NULL) || (start == NULL) || (end == NULL)) {
        return 0.0f;
    }

    // integral
    linalgcuMatrixData_t integral = 0.0f;

    // calc radius
    linalgcuMatrixData_t radius = sqrt(start[0] * start[0]
        + start[1] * start[1]);

    // calc angle
    linalgcuMatrixData_t angleStart = fasteit_basis_angle(x[0], y[0]);
    linalgcuMatrixData_t angleEnd = fasteit_basis_angle(x[1], y[1]) - angleStart;
    linalgcuMatrixData_t angleElectrodeStart = fasteit_basis_angle(start[0], start[1]) - angleStart;
    linalgcuMatrixData_t angleElectrodeEnd = fasteit_basis_angle(end[0], end[1]) - angleStart;

    // correct angle
    angleEnd += (angleEnd < M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeStart += (angleElectrodeStart < M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeEnd += (angleElectrodeEnd < M_PI) ? 2.0f * M_PI : 0.0f;
    angleEnd -= (angleEnd > M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeStart -= (angleElectrodeStart > M_PI) ? 2.0f * M_PI : 0.0f;
    angleElectrodeEnd -= (angleElectrodeEnd > M_PI) ? 2.0f * M_PI : 0.0f;

    // calc parameter
    linalgcuMatrixData_t sEnd = radius * angleEnd;
    linalgcuMatrixData_t sElectrodeStart = radius * angleElectrodeStart;
    linalgcuMatrixData_t sElectrodeEnd = radius * angleElectrodeEnd;

    // integrate left triangle
    if (sEnd < 0.0f) {
        if ((sElectrodeStart < 0.0f) && (sElectrodeEnd > sEnd)) {
            if ((sElectrodeEnd >= 0.0f) && (sElectrodeStart <= sEnd)) {
                integral = -0.5f * sEnd;
            }
            else if ((sElectrodeEnd >= 0.0f) && (sElectrodeStart > sEnd)) {
                integral = -(sElectrodeStart - 0.5 * sElectrodeStart * sElectrodeStart / sEnd);
            }
            else if ((sElectrodeEnd < 0.0f) && (sElectrodeStart <= sEnd)) {
                integral = (sElectrodeEnd - 0.5 * sElectrodeEnd * sElectrodeEnd / sEnd) -
                           (sEnd - 0.5 * sEnd * sEnd / sEnd);
            }
            else if ((sElectrodeEnd < 0.0f) && (sElectrodeStart > sEnd)) {
                integral = (sElectrodeEnd - 0.5 * sElectrodeEnd * sElectrodeEnd / sEnd) -
                           (sElectrodeStart - 0.5 * sElectrodeStart * sElectrodeStart / sEnd);
            }
        }
    }
    else {
        // integrate right triangle
        if ((sElectrodeEnd > 0.0f) && (sEnd > sElectrodeStart)) {
            if ((sElectrodeStart <= 0.0f) && (sElectrodeEnd >= sEnd)) {
                integral = 0.5f * sEnd;
            }
            else if ((sElectrodeStart <= 0.0f) && (sElectrodeEnd < sEnd)) {
                integral = (sElectrodeEnd - 0.5f * sElectrodeEnd * sElectrodeEnd / sEnd);
            }
            else if ((sElectrodeStart > 0.0f) && (sElectrodeEnd >= sEnd)) {
                integral = (sEnd - 0.5f * sEnd * sEnd / sEnd) -
                            (sElectrodeStart - 0.5f * sElectrodeStart * sElectrodeStart / sEnd);
            }
            else if ((sElectrodeStart > 0.0f) && (sElectrodeEnd < sEnd)) {
                integral = (sElectrodeEnd - 0.5f * sElectrodeEnd * sElectrodeEnd / sEnd) -
                            (sElectrodeStart - 0.5f * sElectrodeStart * sElectrodeStart / sEnd);
            }
        }
    }

    return integral;
}
