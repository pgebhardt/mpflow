// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_H
#define FASTEIT_BASIS_H

// basis struct
typedef struct {
    linalgcuMatrixData_t coefficients[3];
    linalgcuMatrixData_t gradient[2];
} fasteitBasis_s;
typedef fasteitBasis_s* fasteitBasis_t;

// create basis
linalgcuError_t fasteit_basis_create(fasteitBasis_t* basisPointer,
    linalgcuMatrixData_t Ax, linalgcuMatrixData_t Ay,
    linalgcuMatrixData_t Bx, linalgcuMatrixData_t By,
    linalgcuMatrixData_t Cx, linalgcuMatrixData_t Cy);

// release basis
linalgcuError_t fasteit_basis_release(fasteitBasis_t* basisPointer);

// evaluate basis function
linalgcuError_t fasteit_basis_function(fasteitBasis_t self,
    linalgcuMatrixData_t* resultPointer, linalgcuMatrixData_t x, linalgcuMatrixData_t y);

#endif
