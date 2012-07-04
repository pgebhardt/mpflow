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
} ert_electrodes_s;
typedef ert_electrodes_s* ert_electrodes_t;

// create electrodes
linalgcl_error_t ert_electrodes_create(ert_electrodes_t* electrodesPointer,
    ert_mesh_t mesh);

// release electrodes
linalgcl_error_t ert_electrodes_release(ert_electrodes_t* electrodesPointer);

#endif
