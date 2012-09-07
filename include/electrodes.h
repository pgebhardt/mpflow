// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_ELECTRODES_H
#define FASTECT_ELECTRODES_H

// electrodes struct
typedef struct {
    linalgcu_size_t count;
    linalgcu_matrix_data_t* electrodesStart;
    linalgcu_matrix_data_t* electrodesEnd;
    linalgcu_matrix_data_t width;
    linalgcu_matrix_data_t height;
} fastect_electrodes_s;
typedef fastect_electrodes_s* fastect_electrodes_t;

// create electrodes
linalgcu_error_t fastect_electrodes_create(fastect_electrodes_t* electrodesPointer,
    linalgcu_size_t count, linalgcu_matrix_data_t width, linalgcu_matrix_data_t height,
    linalgcu_matrix_data_t meshRadius);

// release electrodes
linalgcu_error_t fastect_electrodes_release(fastect_electrodes_t* electrodesPointer);

#endif
