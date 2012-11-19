// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_ELECTRODES_H
#define FASTEIT_ELECTRODES_H

// electrodes struct
typedef struct {
    linalgcuSize_t count;
    linalgcuMatrixData_t* electrodesStart;
    linalgcuMatrixData_t* electrodesEnd;
    linalgcuMatrixData_t width;
    linalgcuMatrixData_t height;
} fasteitElectrodes_s;
typedef fasteitElectrodes_s* fasteitElectrodes_t;

// create electrodes
linalgcuError_t fasteit_electrodes_create(fasteitElectrodes_t* electrodesPointer,
    linalgcuSize_t count, linalgcuMatrixData_t width, linalgcuMatrixData_t height,
    linalgcuMatrixData_t meshRadius);

// release electrodes
linalgcuError_t fasteit_electrodes_release(fasteitElectrodes_t* electrodesPointer);

#endif
