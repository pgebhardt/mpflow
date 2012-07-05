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

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <linalgcl/linalgcl.h>
#include "basis.h"
#include "mesh.h"
#include "electrodes.h"

// create electrodes
linalgcl_error_t ert_electrodes_create(ert_electrodes_t* electrodesPointer,
    linalgcl_size_t count, linalgcl_matrix_data_t size, ert_mesh_t mesh) {
    // check input
    if ((electrodesPointer == NULL) || (count == 0) || (mesh == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    // create struct
    ert_electrodes_t electrodes = malloc(sizeof(ert_electrodes_s));

    // check success
    if (electrodes == NULL) {
        return LINALGCL_ERROR;
    }

    // init struct
    electrodes->count = count;
    electrodes->electrode_start = NULL;
    electrodes->electrode_end = NULL;

    // create electrode vectors
    electrodes->electrode_start = malloc(sizeof(linalgcl_matrix_data_t) *
        electrodes->count * 2);
    electrodes->electrode_end = malloc(sizeof(linalgcl_matrix_data_t) *
        electrodes->count * 2);

    // check success
    if ((electrodes->electrode_start == NULL) || (electrodes->electrode_end == NULL)) {
        // cleanup
        ert_electrodes_release(&electrodes);

        return LINALGCL_ERROR;
    }

    // fill electrode vectors
    linalgcl_matrix_data_t angle = 0.0f;
    linalgcl_matrix_data_t delta_angle = M_PI / (linalgcl_matrix_data_t)electrodes->count;
    printf("delta_angle: %f\n", delta_angle);
    for (linalgcl_size_t i = 0; i < electrodes->count; i++) {
        // calc start angle
        angle = (linalgcl_matrix_data_t)i * 2.0f * delta_angle;

        // calc start coordinates
        electrodes->electrode_start[i * 2 + 0] = mesh->radius * cos(angle);
        electrodes->electrode_start[i * 2 + 1] = mesh->radius * sin(angle);

        // calc end angle
        angle += delta_angle;

        // calc end coordinates
        electrodes->electrode_end[i * 2 + 0] = mesh->radius * cos(angle);
        electrodes->electrode_end[i * 2 + 1] = mesh->radius * sin(angle);
    }

    // set electrodesPointer
    *electrodesPointer = electrodes;

    return LINALGCL_SUCCESS;
}

// release electrodes
linalgcl_error_t ert_electrodes_release(ert_electrodes_t* electrodesPointer) {
    // check input
    if ((electrodesPointer == NULL) || (*electrodesPointer == NULL)) {
        return LINALGCL_ERROR;
    }

    // get electrodes
    ert_electrodes_t electrodes = *electrodesPointer;

    // free electrode vectors
    if (electrodes->electrode_start != NULL) {
        free(electrodes->electrode_start);
    }
    if (electrodes->electrode_end != NULL) {
        free(electrodes->electrode_end);
    }

    // free struct
    free(electrodes);

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    return LINALGCL_SUCCESS;
}
