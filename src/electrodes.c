// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fastect.h"

// create electrodes
linalgcuError_t fastect_electrodes_create(fastectElectrodes_t* electrodesPointer,
    linalgcuSize_t count, linalgcuMatrixData_t width, linalgcuMatrixData_t height,
    linalgcuMatrixData_t meshRadius) {
    // check input
    if ((electrodesPointer == NULL) || (count == 0) || (width <= 0.0f) || (height <= 0.0f) ||
        (meshRadius <= 0.0f)) {
        return LINALGCU_ERROR;
    }

    // error
    linalgcuError_t error = LINALGCU_SUCCESS;

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    // create struct
    fastectElectrodes_t electrodes = malloc(sizeof(fastectElectrodes_s));

    // check success
    if (electrodes == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    electrodes->count = count;
    electrodes->electrodesStart = NULL;
    electrodes->electrodesEnd = NULL;
    electrodes->width = width;
    electrodes->height = height;

    // create electrode vectors
    electrodes->electrodesStart = malloc(sizeof(linalgcuMatrixData_t) *
        electrodes->count * 2);
    electrodes->electrodesEnd = malloc(sizeof(linalgcuMatrixData_t) *
        electrodes->count * 2);

    // check success
    if ((electrodes->electrodesStart == NULL) || (electrodes->electrodesEnd == NULL)) {
        // cleanup
        fastect_electrodes_release(&electrodes);

        return LINALGCU_ERROR;
    }

    // fill electrode vectors
    linalgcuMatrixData_t angle = 0.0f;
    linalgcuMatrixData_t delta_angle = M_PI / (linalgcuMatrixData_t)electrodes->count;
    for (linalgcuSize_t i = 0; i < electrodes->count; i++) {
        // calc start angle
        angle = (linalgcuMatrixData_t)i * 2.0f * delta_angle;

        // calc start coordinates
        electrodes->electrodesStart[i * 2 + 0] = meshRadius * cos(angle);
        electrodes->electrodesStart[i * 2 + 1] = meshRadius * sin(angle);

        // calc end angle
        angle += electrodes->width / meshRadius;

        // calc end coordinates
        electrodes->electrodesEnd[i * 2 + 0] = meshRadius * cos(angle);
        electrodes->electrodesEnd[i * 2 + 1] = meshRadius * sin(angle);
    }

    // set electrodesPointer
    *electrodesPointer = electrodes;

    return LINALGCU_SUCCESS;
}

// release electrodes
linalgcuError_t fastect_electrodes_release(fastectElectrodes_t* electrodesPointer) {
    // check input
    if ((electrodesPointer == NULL) || (*electrodesPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get electrodes
    fastectElectrodes_t electrodes = *electrodesPointer;

    // free electrode vectors
    if (electrodes->electrodesStart != NULL) {
        free(electrodes->electrodesStart);
    }
    if (electrodes->electrodesEnd != NULL) {
        free(electrodes->electrodesEnd);
    }

    // free struct
    free(electrodes);

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    return LINALGCU_SUCCESS;
}
