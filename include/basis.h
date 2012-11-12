// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_BASIS_H
#define FASTEIT_BASIS_H

// nodes per element
#define FASTEIT_NODES_PER_ELEMENT (3)
#define FASTEIT_NODES_PER_EDGE (2)

// basis struct
typedef struct {
    linalgcuMatrixData_t points[FASTEIT_NODES_PER_ELEMENT][2];
    linalgcuMatrixData_t coefficients[FASTEIT_NODES_PER_ELEMENT];
} fasteitBasis_s;
typedef fasteitBasis_s* fasteitBasis_t;

// create basis
linalgcuError_t fasteit_basis_create(fasteitBasis_t* basisPointer,
    linalgcuMatrixData_t* nodeX, linalgcuMatrixData_t* nodeY);

// release basis
linalgcuError_t fasteit_basis_release(fasteitBasis_t* basisPointer);

// evaluate basis function
linalgcuError_t fasteit_basis_function(fasteitBasis_t self,
    linalgcuMatrixData_t* resultPointer, linalgcuMatrixData_t x, linalgcuMatrixData_t y);

// integrate with basis
linalgcuMatrixData_t fasteit_basis_integrate_with_basis(fasteitBasis_t self, fasteitBasis_t other);

// integrate gradient with basis
linalgcuMatrixData_t fasteit_basis_integrate_gradient_with_basis(fasteitBasis_t self,
    fasteitBasis_t other);

#endif
