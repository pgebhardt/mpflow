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
    linalgcl_size_t count) {
    // check input
    if ((electrodesPointer == NULL) || (count == 0)) {
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
    electrodes->electrode_vertices = NULL;

    // create electrode vertices memory
    electrodes->electrode_vertices = malloc(sizeof(linalgcl_matrix_s) *
        electrodes->count);

    // check success
    if (electrodes->electrode_vertices == NULL) {
        // cleanup
        ert_electrodes_release(&electrodes);

        return LINALGCL_ERROR;
    }

    // init matrix pointer to NULL
    for (linalgcl_size_t i = 0; i < electrodes->count; i++) {
        electrodes->electrode_vertices[i] = NULL;
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

    // release electrode vertices
    for (linalgcl_size_t i = 0; i < electrodes->count; i++) {
        linalgcl_matrix_release(&electrodes->electrode_vertices[i]);
    }
    free(electrodes->electrode_vertices);

    // free struct
    free(electrodes);

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    return LINALGCL_SUCCESS;
}

linalgcl_matrix_data_t ert_electrodes_angle(linalgcl_matrix_data_t x,
    linalgcl_matrix_data_t y) {
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

// get vertices for electrodes
linalgcl_error_t ert_electrodes_get_vertices(ert_electrodes_t electrodes,
    ert_mesh_t mesh, cl_context context) {
    // check input
    if ((electrodes == NULL) || (mesh == NULL) || (context == NULL)) {
        return LINALGCL_ERROR;
    }

    // error
    linalgcl_error_t error = LINALGCL_SUCCESS;

    // delta angle for electrodes
    linalgcl_matrix_data_t delta_angle = 2.0f * M_PI / (2.0f * (linalgcl_matrix_data_t)electrodes->count);

    // get vertex per electrode count for first electrode
    linalgcl_size_t vertex_count = 0;
    linalgcl_matrix_data_t angle, radius;
    linalgcl_matrix_data_t x, y;

    for (linalgcl_size_t i = 0; i < mesh->vertex_count; i++) {
        // get vertex
        linalgcl_matrix_get_element(mesh->vertices, &x, i, 0);
        linalgcl_matrix_get_element(mesh->vertices, &y, i, 1);

        // calc radius and angle
        radius = sqrt(x * x + y * y);
        angle = ert_electrodes_angle(x, y);

        // check radius and angle
        if ((radius >= mesh->radius - mesh->distance / 4.0f) && (angle >= 0.0f) && (angle <= delta_angle)) {
            vertex_count++;
        }
    }

    // create matrices
    for (linalgcl_size_t i = 0; i < electrodes->count; i++) {
        error += linalgcl_matrix_create(&electrodes->electrode_vertices[i], context, vertex_count * 2, 1);
    }

    // check success
    if (error != LINALGCL_SUCCESS) {
        return error;
    }

    // fill matrices
    linalgcl_size_t electrode;
    linalgcl_matrix_data_t id;

    linalgcl_size_t* count = malloc(sizeof(linalgcl_size_t) * electrodes->count);
    for (linalgcl_size_t i = 0; i < electrodes->count; i++) {
        count[i] = 0;
    }

    for (linalgcl_size_t i = 0; i < mesh->vertex_count; i++) {
        // get vertex
        linalgcl_matrix_get_element(mesh->vertices, &x, i, 0);
        linalgcl_matrix_get_element(mesh->vertices, &y, i, 1);

        // calc radius and angle
        radius = sqrt(x * x + y * y);
        angle = ert_electrodes_angle(x, y);
        angle += angle < 0.0f ? 2.0f * M_PI : 0.0f;

        // check radius and angle
        if ((radius >= mesh->radius - mesh->distance / 4.0f)) {
            electrode = (linalgcl_size_t)(angle / delta_angle);

            if (electrode % 2 == 0) {
                // get electrode id
                id = (linalgcl_matrix_data_t)i;

                linalgcl_matrix_set_element(electrodes->electrode_vertices[electrode / 2],
                    id, count[electrode / 2], 0);

                count[electrode / 2]++;
            }
        }
    }

    // cleanup
    free(count);

    return LINALGCL_SUCCESS;
}
