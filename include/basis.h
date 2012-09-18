// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_BASIS_H
#define FASTECT_BASIS_H

// basis struct
typedef struct {
    linalgcuMatrixData_t coefficients[3];
    linalgcuMatrixData_t gradient[2];
} fastectBasts_s;
typedef fastectBasts_s* fastectBasis_t;

// create basis
linalgcuError_t fastect_basis_create(fastectBasis_t* basisPointer,
    linalgcuMatrixData_t Ax, linalgcuMatrixData_t Ay,
    linalgcuMatrixData_t Bx, linalgcuMatrixData_t By,
    linalgcuMatrixData_t Cx, linalgcuMatrixData_t Cy);

// release basis
linalgcuError_t fastect_basis_release(fastectBasis_t* basisPointer);

// evaluate basis function
linalgcuError_t fastect_basis_function(fastectBasis_t basis,
    linalgcuMatrixData_t* resultPointer, linalgcuMatrixData_t x, linalgcuMatrixData_t y);

#endif
