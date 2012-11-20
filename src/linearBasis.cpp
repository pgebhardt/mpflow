// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace std;

// create basis class
Basis::Basis(linalgcuMatrixData_t* x, linalgcuMatrixData_t* y) {
    // check input
    if (x == NULL) {
        throw invalid_argument("x == NULL");
    }
    if (y == NULL) {
        throw invalid_argument("y == NULL");
    }

    // create memory
    this->mPoints = new linalgcuMatrixData_t[Basis::nodesPerElement * 2];
    this->mCoefficients = new linalgcuMatrixData_t[Basis::nodesPerElement];

    // init member
    for (linalgcuSize_t i = 0; i < Basis::nodesPerElement; i++) {
        this->mPoints[i * 2 + 0] = x[i];
        this->mPoints[i * 2 + 1] = y[i];
        this->mCoefficients[i] = 0.0;
    }

    // calc coefficients (A * c = b)
    linalgcuMatrixData_t Ainv[3][3];
    linalgcuMatrixData_t B[3] = {1.0, 0.0, 0.0};

    // invert matrix A directly
    linalgcuMatrixData_t a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = this->mPoints[0 * 2 + 0];
    c = this->mPoints[0 * 2 + 1];
    d = 1.0;
    e = this->mPoints[1 * 2 + 0];
    f = this->mPoints[1 * 2 + 1];
    g = 1.0;
    h = this->mPoints[2 * 2 + 0];
    i = this->mPoints[2 * 2 + 1];
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
    this->mCoefficients[0] = Ainv[0][0] * B[0] + Ainv[0][1] * B[1] + Ainv[0][2] * B[2];
    this->mCoefficients[1] = Ainv[1][0] * B[0] + Ainv[1][1] * B[1] + Ainv[1][2] * B[2];
    this->mCoefficients[2] = Ainv[2][0] * B[0] + Ainv[2][1] * B[1] + Ainv[2][2] * B[2];
}

// delete basis class
Basis::~Basis() {
    // cleanup arrays
    if (this->mPoints != NULL) {
        delete [] this->mPoints;
    }
    if (this->mCoefficients != NULL) {
        delete [] this->mCoefficients;
    }
}

// evaluate basis function
linalgcuMatrixData_t Basis::evaluate(linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    // calc result
    return this->mCoefficients[0] + this->mCoefficients[1] * x +
        this->mCoefficients[2] * y;
}

// integrate with basis
linalgcuMatrixData_t Basis::integrate_with_basis(Basis& other) {
    // shorten variables
    linalgcuMatrixData_t x1 = this->mPoints[0 * 2 + 0];
    linalgcuMatrixData_t y1 = this->mPoints[0 * 2 + 1];
    linalgcuMatrixData_t x2 = this->mPoints[1 * 2 + 0];
    linalgcuMatrixData_t y2 = this->mPoints[1 * 2 + 1];
    linalgcuMatrixData_t x3 = this->mPoints[2 * 2 + 0];
    linalgcuMatrixData_t y3 = this->mPoints[2 * 2 + 1];

    linalgcuMatrixData_t ai = this->mCoefficients[0];
    linalgcuMatrixData_t bi = this->mCoefficients[1];
    linalgcuMatrixData_t ci = this->mCoefficients[2];
    linalgcuMatrixData_t aj = other.coefficient(0);
    linalgcuMatrixData_t bj = other.coefficient(1);
    linalgcuMatrixData_t cj = other.coefficient(2);

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
linalgcuMatrixData_t Basis::integrate_gradient_with_basis(Basis& other) {
    // calc area
    linalgcuMatrixData_t area = 0.5 * fabs((this->mPoints[1 * 2 + 0] - this->mPoints[0 * 2 + 0]) *
        (this->mPoints[2 * 2 + 1] - this->mPoints[0 * 2 + 1]) -
        (this->mPoints[2 * 2 + 0] - this->mPoints[0 * 2 + 0]) *
        (this->mPoints[1 * 2 + 1] - this->mPoints[0 * 2 + 1]));

    // calc integral
    return area * (this->mCoefficients[1] * other.coefficient(1) +
        this->mCoefficients[2] * other.coefficient(2));
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

// calc parametrisation
linalgcuMatrixData_t fasteit_basis_calc_parametrisation(linalgcuMatrixData_t x, linalgcuMatrixData_t y,
    linalgcuMatrixData_t offset) {
    // convert to polar coordinates
    linalgcuMatrixData_t radius = sqrt(x * x + y * y);
    linalgcuMatrixData_t angle = fasteit_basis_angle(x, y) - offset / radius;

    // correct angle
    angle += (angle < M_PI) ? 2.0f * M_PI : 0.0f;
    angle -= (angle > M_PI) ? 2.0f * M_PI : 0.0f;

    // calc parameter
    linalgcuMatrixData_t parameter = angle * radius;

    return parameter;
}

// integrate edge
linalgcuMatrixData_t Basis::integrate_boundary_edge(linalgcuMatrixData_t* x,
    linalgcuMatrixData_t* y, linalgcuMatrixData_t* start, linalgcuMatrixData_t* end) {
    // check input
    if ((x == NULL) || (y == NULL) || (start == NULL) || (end == NULL)) {
        return 0.0f;
    }

    // integral
    linalgcuMatrixData_t integral = 0.0f;

    // calc node parameter
    linalgcuMatrixData_t* nodeParameter = new linalgcuMatrixData_t[Basis::nodesPerEdge];
    nodeParameter[0] = 0.0f;
    for (linalgcuSize_t i = 0; i < Basis::nodesPerEdge; i++) {
        nodeParameter[i] = fasteit_basis_calc_parametrisation(x[i], y[i], nodeParameter[0]);
    }

    // calc integration boundary parameter
    linalgcuMatrixData_t boundaryParameter[2];
    boundaryParameter[0] = fasteit_basis_calc_parametrisation(start[0], start[1], nodeParameter[0]);
    boundaryParameter[1] = fasteit_basis_calc_parametrisation(end[0], end[1], nodeParameter[0]);

    // integrate left triangle
    if (nodeParameter[1] < 0.0f) {
        if ((boundaryParameter[0] < 0.0f) && (boundaryParameter[1] > nodeParameter[1])) {
            if ((boundaryParameter[1] >= 0.0f) && (boundaryParameter[0] <= nodeParameter[1])) {
                integral = -0.5f * nodeParameter[1];
            }
            else if ((boundaryParameter[1] >= 0.0f) && (boundaryParameter[0] > nodeParameter[1])) {
                integral = -(boundaryParameter[0] - 0.5 * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);
            }
            else if ((boundaryParameter[1] < 0.0f) && (boundaryParameter[0] <= nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5 * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]) -
                           (nodeParameter[1] - 0.5 * nodeParameter[1] * nodeParameter[1] / nodeParameter[1]);
            }
            else if ((boundaryParameter[1] < 0.0f) && (boundaryParameter[0] > nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5 * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]) -
                           (boundaryParameter[0] - 0.5 * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);
            }
        }
    }
    else {
        // integrate right triangle
        if ((boundaryParameter[1] > 0.0f) && (nodeParameter[1] > boundaryParameter[0])) {
            if ((boundaryParameter[0] <= 0.0f) && (boundaryParameter[1] >= nodeParameter[1])) {
                integral = 0.5f * nodeParameter[1];
            }
            else if ((boundaryParameter[0] <= 0.0f) && (boundaryParameter[1] < nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5f * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]);
            }
            else if ((boundaryParameter[0] > 0.0f) && (boundaryParameter[1] >= nodeParameter[1])) {
                integral = (nodeParameter[1] - 0.5f * nodeParameter[1] * nodeParameter[1] / nodeParameter[1]) -
                            (boundaryParameter[0] - 0.5f * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);
            }
            else if ((boundaryParameter[0] > 0.0f) && (boundaryParameter[1] < nodeParameter[1])) {
                integral = (boundaryParameter[1] - 0.5f * boundaryParameter[1] * boundaryParameter[1] / nodeParameter[1]) -
                            (boundaryParameter[0] - 0.5f * boundaryParameter[0] * boundaryParameter[0] / nodeParameter[1]);
            }
        }
    }

    // cleanup
    delete [] nodeParameter;

    return integral;

}

// operator
linalgcuMatrixData_t Basis::operator() (linalgcuMatrixData_t x, linalgcuMatrixData_t y) {
    return this->evaluate(x, y);
}
