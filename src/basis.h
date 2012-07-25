// ert
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef ERT_BASIS_H
#define ERT_BASIS_H

// basis struct
typedef struct {
    linalgcu_matrix_data_t coefficients[3];
    linalgcu_matrix_data_t gradient[2];
} ert_basis_s;
typedef ert_basis_s* ert_basis_t;

// create basis
linalgcu_error_t ert_basis_create(ert_basis_t* basisPointer,
    linalgcu_matrix_data_t Ax, linalgcu_matrix_data_t Ay,
    linalgcu_matrix_data_t Bx, linalgcu_matrix_data_t By,
    linalgcu_matrix_data_t Cx, linalgcu_matrix_data_t Cy);

// release basis
linalgcu_error_t ert_basis_release(ert_basis_t* basisPointer);

// evaluate basis function
linalgcu_error_t ert_basis_function(ert_basis_t basis, linalgcu_matrix_data_t* resultPointer,
    linalgcu_matrix_data_t x, linalgcu_matrix_data_t y);

#endif
