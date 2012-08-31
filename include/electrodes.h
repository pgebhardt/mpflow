// fastECT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTECT_ELECTRODES_H
#define FASTECT_ELECTRODES_H

// c++ compatibility
#ifdef __cplusplus
extern "C" {
#endif

// electrodes struct
typedef struct {
    linalgcu_size_t count;
    linalgcu_matrix_data_t* electrode_start;
    linalgcu_matrix_data_t* electrode_end;
    linalgcu_matrix_data_t size;
} fastect_electrodes_s;
typedef fastect_electrodes_s* fastect_electrodes_t;

// create electrodes
linalgcu_error_t fastect_electrodes_create(fastect_electrodes_t* electrodesPointer,
    linalgcu_size_t count, linalgcu_matrix_data_t size, linalgcu_matrix_data_t radius);

// release electrodes
linalgcu_error_t fastect_electrodes_release(fastect_electrodes_t* electrodesPointer);

#ifdef __cplusplus
}
#endif

#endif
