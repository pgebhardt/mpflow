// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_ELECTRODES_H
#define FASTECT_ELECTRODES_H

// electrodes struct
typedef struct {
    linalgcuSize_t count;
    linalgcuMatrixData_t* electrodesStart;
    linalgcuMatrixData_t* electrodesEnd;
    linalgcuMatrixData_t width;
    linalgcuMatrixData_t height;
} fastectElectrodes_s;
typedef fastectElectrodes_s* fastectElectrodes_t;

// create electrodes
linalgcuError_t fastect_electrodes_create(fastectElectrodes_t* electrodesPointer,
    linalgcuSize_t count, linalgcuMatrixData_t width, linalgcuMatrixData_t height,
    linalgcuMatrixData_t meshRadius);

// release electrodes
linalgcuError_t fastect_electrodes_release(fastectElectrodes_t* electrodesPointer);

#endif
