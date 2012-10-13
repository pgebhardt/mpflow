// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_FASTEIT_H
#define FASTEIT_FASTEIT_H

#include <cuda/cuda_runtime.h>
#include <cuda/cublas_v2.h>
#include <linalgcu/linalgcu.h>

// c++ compatibility
#ifdef __cplusplus
extern "C" {
#endif

#include "mesh.h"
#include "basis.h"
#include "electrodes.h"
#include "grid.h"
#include "conjugate.h"
#include "conjugate_sparse.h"
#include "forward.h"
#include "inverse.h"
#include "solver.h"

#ifdef __cplusplus
}
#endif

#endif
