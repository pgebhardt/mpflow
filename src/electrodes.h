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

#ifndef ERT_ELECTRODES_H
#define ERT_ELECTRODES_H

// electrodes struct
typedef struct {
    linalgcl_size_t count;
    linalgcl_matrix_t* electrode_vertices;
    linalgcl_size_t* vertex_count;
} ert_electrodes_s;
typedef ert_electrodes_s* ert_electrodes_t;

// create electrodes
linalgcl_error_t ert_electrodes_create(ert_electrodes_t* electrodesPointer,
    linalgcl_size_t count);

// release electrodes
linalgcl_error_t ert_electrodes_release(ert_electrodes_t* electrodesPointer);

// get vertices for electrodes
linalgcl_error_t ert_electrodes_get_vertices(ert_electrodes_t electrodes,
    ert_mesh_t mesh, cl_context context);

#endif
