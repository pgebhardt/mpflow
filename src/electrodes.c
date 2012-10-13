// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <stdlib.h>
#include "../include/fasteit.h"

// create electrodes
linalgcuError_t fasteit_electrodes_create(fasteitElectrodes_t* electrodesPointer,
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
    fasteitElectrodes_t self = malloc(sizeof(fasteitElectrodes_s));

    // check success
    if (self == NULL) {
        return LINALGCU_ERROR;
    }

    // init struct
    self->count = count;
    self->electrodesStart = NULL;
    self->electrodesEnd = NULL;
    self->width = width;
    self->height = height;

    // create electrode vectors
    self->electrodesStart = malloc(sizeof(linalgcuMatrixData_t) *
        self->count * 2);
    self->electrodesEnd = malloc(sizeof(linalgcuMatrixData_t) *
        self->count * 2);

    // check success
    if ((self->electrodesStart == NULL) || (self->electrodesEnd == NULL)) {
        // cleanup
        fasteit_electrodes_release(&self);

        return LINALGCU_ERROR;
    }

    // fill electrode vectors
    linalgcuMatrixData_t angle = 0.0f;
    linalgcuMatrixData_t delta_angle = M_PI / (linalgcuMatrixData_t)self->count;
    for (linalgcuSize_t i = 0; i < self->count; i++) {
        // calc start angle
        angle = (linalgcuMatrixData_t)i * 2.0f * delta_angle;

        // calc start coordinates
        self->electrodesStart[i * 2 + 0] = meshRadius * cos(angle);
        self->electrodesStart[i * 2 + 1] = meshRadius * sin(angle);

        // calc end angle
        angle += self->width / meshRadius;

        // calc end coordinates
        self->electrodesEnd[i * 2 + 0] = meshRadius * cos(angle);
        self->electrodesEnd[i * 2 + 1] = meshRadius * sin(angle);
    }

    // set electrodesPointer
    *electrodesPointer = self;

    return LINALGCU_SUCCESS;
}

// release electrodes
linalgcuError_t fasteit_electrodes_release(fasteitElectrodes_t* electrodesPointer) {
    // check input
    if ((electrodesPointer == NULL) || (*electrodesPointer == NULL)) {
        return LINALGCU_ERROR;
    }

    // get electrodes
    fasteitElectrodes_t self = *electrodesPointer;

    // free electrode vectors
    if (self->electrodesStart != NULL) {
        free(self->electrodesStart);
    }
    if (self->electrodesEnd != NULL) {
        free(self->electrodesEnd);
    }

    // free struct
    free(self);

    // set electrodesPointer to NULL
    *electrodesPointer = NULL;

    return LINALGCU_SUCCESS;
}
