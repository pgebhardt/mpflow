// fastECT
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
#include <math.h>
#include "fastect.h"

// create electrodes
linalgcu_error_t fastect_electrodes_create(fastect_electrodes_t* electrodesPointer,
    linalgcu_size_t count, linalgcu_matrix_data_t size, fastect_mesh_t mesh) {
    // check input
    if ((electrodesPointer == NULL) || (count == 0) || (mesh == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    // create struct
    fastect_electrodes_t electrodes = malloc(sizeof(fastect_electrodes_s));

    // check success
    if (electrodes == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    electrodes->count = count;
    electrodes->electrode_start = NULL;
    electrodes->electrode_end = NULL;
    electrodes->size = size;

    // create electrode vectors
    electrodes->electrode_start = malloc(sizeof(linalgcu_matrix_data_t) *
        electrodes->count * 2);
    electrodes->electrode_end = malloc(sizeof(linalgcu_matrix_data_t) *
        electrodes->count * 2);

    // check success
    if ((electrodes->electrode_start == NULL) || (electrodes->electrode_end == NULL)) {
        // cleanup
        fastect_electrodes_release(&electrodes);

        return LINALGCU_ERROR;
    }

    // fill electrode vectors
    linalgcu_matrix_data_t angle = 0.0f;
    linalgcu_matrix_data_t delta_angle = M_PI / (linalgcu_matrix_data_t)electrodes->count;
    for (linalgcu_size_t i = 0; i < electrodes->count; i++) {
        // calc start angle
        angle = (linalgcu_matrix_data_t)i * 2.0f * delta_angle;

        // calc start coordinates
        electrodes->electrode_start[i * 2 + 0] = mesh->radius * cos(angle);
        electrodes->electrode_start[i * 2 + 1] = mesh->radius * sin(angle);

        // calc end angle
        angle += size / mesh->radius;

        // calc end coordinates
        electrodes->electrode_end[i * 2 + 0] = mesh->radius * cos(angle);
        electrodes->electrode_end[i * 2 + 1] = mesh->radius * sin(angle);
    }

    // set electrodesPointer
    *electrodesPointer = electrodes;

    return LINALGCU_SUCCESS;
}

// create new electrodes from config
linalgcu_error_t fastect_electrodes_create_from_config(fastect_electrodes_t* electrodesPointer,
    config_setting_t* settings, fastect_mesh_t mesh) {
    // check input
    if ((electrodesPointer == NULL) || (settings == NULL) || (mesh == NULL)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcu_error_t error = LINALGCU_SUCCESS;

    // reset electrodes pointer
    *electrodesPointer = NULL;

    // lookup settings
    int electrodes_count;
    double electrodes_size;
    if (!(config_setting_lookup_int(settings, "count", &electrodes_count) &&
        config_setting_lookup_float(settings, "size", &electrodes_size))) {
        return LINALGCU_ERROR;
    }

    // create electrodes
    fastect_electrodes_t electrodes = NULL;
    error = fastect_electrodes_create(&electrodes, electrodes_count, electrodes_size, mesh);

    // check success
    if (error != LINALGCU_SUCCESS) {
        return error;
    }

    // set electrodes pointer
    *electrodesPointer = electrodes;

    return LINALGCU_SUCCESS;
}

// release electrodes
linalgcu_error_t fastect_electrodes_release(fastect_electrodes_t* electrodesPointer) {
    // check input
    if ((electrodesPointer == NULL) || (*electrodesPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get electrodes
    fastect_electrodes_t electrodes = *electrodesPointer;

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

    return LINALGCU_SUCCESS;
}
