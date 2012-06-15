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

#ifndef ERT_SOLVER_H
#define ERT_SOLVER_H

// solver struct
typedef struct {
    ert_mesh_t mesh;
    linalg_matrix_t A;
    linalg_matrix_t B;
} ert_solver_s;
typedef ert_solver_s* ert_solver_t;

// create solver
linalg_error_t ert_solver_create(ert_solver_t* solverPointer, ert_mesh_t mesh);

// release solver
linalg_error_t ert_solver_release(ert_solver_t* solverPointer);

#endif
