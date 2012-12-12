// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FASTEIT_H
#define FASTEIT_INCLUDE_FASTEIT_H

// std lib includes
#include <assert.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <array>
#include <vector>
#include <memory>
#include <algorithm>

// cuda includes
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "dtype.h"
#include "constants.h"
#include "math.h"
#include "matrix.h"
#include "sparse_matrix.h"
#include "basis.h"
#include "mesh.h"
#include "electrodes.h"
#include "model.h"
#include "conjugate.h"
#include "sparse_conjugate.h"
#include "forward.h"
#include "inverse.h"
#include "solver.h"

#endif
