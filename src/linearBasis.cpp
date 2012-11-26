// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include "../include/fasteit.hpp"

// namespaces
using namespace fastEIT;
using namespace fastEIT::basis;
using namespace std;

// create basis class
Linear::Linear(dtype::real* x, dtype::real* y) {
    // check input
    if (x == NULL) {
        throw invalid_argument("x == NULL");
    }
    if (y == NULL) {
        throw invalid_argument("y == NULL");
    }

    // create memory
    this->mPoints = new dtype::real[Linear::nodesPerElement * 2];
    this->mCoefficients = new dtype::real[Linear::nodesPerElement];

    // init member
    for (dtype::size i = 0; i < Linear::nodesPerElement; i++) {
        this->mPoints[i * 2 + 0] = x[i];
        this->mPoints[i * 2 + 1] = y[i];
        this->mCoefficients[i] = 0.0;
    }

    // calc coefficients (A * c = b)
    dtype::real Ainv[3][3];
    dtype::real B[3] = {1.0, 0.0, 0.0};

    // invert matrix A directly
    dtype::real a, b, c, d, e, f, g, h, i;
    a = 1.0;
    b = this->mPoints[0 * 2 + 0];
    c = this->mPoints[0 * 2 + 1];
    d = 1.0;
    e = this->mPoints[1 * 2 + 0];
    f = this->mPoints[1 * 2 + 1];
    g = 1.0;
    h = this->mPoints[2 * 2 + 0];
    i = this->mPoints[2 * 2 + 1];
    dtype::real det = a * (e * i - f * h) - b * (i * d - f * g) + c * (d * h - e * g);

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
Linear::~Linear() {
    // cleanup arrays
    delete [] this->mPoints;
    delete [] this->mCoefficients;
}

// evaluate basis function
dtype::real Linear::operator() (dtype::real x, dtype::real y) {
    // calc result
    return this->mCoefficients[0] + this->mCoefficients[1] * x +
        this->mCoefficients[2] * y;
}

// integrate with basis
dtype::real Linear::integrateWithBasis(Linear& other) {
    // shorten variables
    dtype::real x1 = this->mPoints[0 * 2 + 0];
    dtype::real y1 = this->mPoints[0 * 2 + 1];
    dtype::real x2 = this->mPoints[1 * 2 + 0];
    dtype::real y2 = this->mPoints[1 * 2 + 1];
    dtype::real x3 = this->mPoints[2 * 2 + 0];
    dtype::real y3 = this->mPoints[2 * 2 + 1];

    dtype::real ai = this->mCoefficients[0];
    dtype::real bi = this->mCoefficients[1];
    dtype::real ci = this->mCoefficients[2];
    dtype::real aj = other.coefficient(0);
    dtype::real bj = other.coefficient(1);
    dtype::real cj = other.coefficient(2);

    // calc area
    dtype::real area = 0.5 * fabs((x2 - x1) * (y3 - y1) -
        (x3 - x1) * (y2 - y1));

    // calc integral
    dtype::real integral = 2.0f * area *
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
dtype::real Linear::integrateGradientWithBasis(Linear& other) {
    // calc area
    dtype::real area = 0.5 * fabs((this->mPoints[1 * 2 + 0] - this->mPoints[0 * 2 + 0]) *
        (this->mPoints[2 * 2 + 1] - this->mPoints[0 * 2 + 1]) -
        (this->mPoints[2 * 2 + 0] - this->mPoints[0 * 2 + 0]) *
        (this->mPoints[1 * 2 + 1] - this->mPoints[0 * 2 + 1]));

    // calc integral
    return area * (this->mCoefficients[1] * other.coefficient(1) +
        this->mCoefficients[2] * other.coefficient(2));
}

// angle calculation for parametrisation
dtype::real fasteit_basis_angle(dtype::real x, dtype::real y) {
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
dtype::real fasteit_basis_calc_parametrisation(dtype::real x, dtype::real y,
    dtype::real offset) {
    // convert to polar coordinates
    dtype::real radius = sqrt(x * x + y * y);
    dtype::real angle = fasteit_basis_angle(x, y) - offset / radius;

    // correct angle
    angle += (angle < M_PI) ? 2.0f * M_PI : 0.0f;
    angle -= (angle > M_PI) ? 2.0f * M_PI : 0.0f;

    // calc parameter
    dtype::real parameter = angle * radius;

    return parameter;
}

// integrate edge
dtype::real Linear::integrateBoundaryEdge(dtype::real* x,
    dtype::real* y, dtype::real* start, dtype::real* end) {
    // check input
    if ((x == NULL) || (y == NULL) || (start == NULL) || (end == NULL)) {
        return 0.0f;
    }

    // integral
    dtype::real integral = 0.0f;

    // calc node parameter
    dtype::real* nodeParameter = new dtype::real[Linear::nodesPerEdge];
    nodeParameter[0] = 0.0f;
    for (dtype::size i = 0; i < Linear::nodesPerEdge; i++) {
        nodeParameter[i] = fasteit_basis_calc_parametrisation(x[i], y[i], nodeParameter[0]);
    }

    // calc integration boundary parameter
    dtype::real boundaryParameter[2];
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
