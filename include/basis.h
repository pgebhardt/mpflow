// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_BASIS_H
#define FASTECT_BASIS_H

// c++ compatibility
#ifdef __cplusplus
extern "C" {
#endif

// basis struct
typedef struct {
    linalgcu_matrix_data_t coefficients[3];
    linalgcu_matrix_data_t gradient[2];
} fastect_basis_s;
typedef fastect_basis_s* fastect_basis_t;

// create basis
linalgcu_error_t fastect_basis_create(fastect_basis_t* basisPointer,
    linalgcu_matrix_data_t Ax, linalgcu_matrix_data_t Ay,
    linalgcu_matrix_data_t Bx, linalgcu_matrix_data_t By,
    linalgcu_matrix_data_t Cx, linalgcu_matrix_data_t Cy);

// release basis
linalgcu_error_t fastect_basis_release(fastect_basis_t* basisPointer);

// evaluate basis function
linalgcu_error_t fastect_basis_function(fastect_basis_t basis,
    linalgcu_matrix_data_t* resultPointer, linalgcu_matrix_data_t x, linalgcu_matrix_data_t y);

#ifdef __cplusplus
}
#endif

#endif
