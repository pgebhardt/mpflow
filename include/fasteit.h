// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FASTEIT_H
#define FASTEIT_INCLUDE_FASTEIT_H

#include <cstdlib>
#include <cstdio>

// std lib includes
#include <assert.h>
#include <stdexcept>
#include <string>
#include <istream>
#include <iostream>
#include <tuple>
#include <array>
#include <vector>
#include <memory>
#include <algorithm>

// cuda includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda_error.h"

#include "dtype.h"
#include "constants.h"
#include "math.h"
#include "matrix.h"
#include "sparse_matrix.h"
#include "basis.h"
#include "mesh.h"
#include "electrodes.h"
#include "source.h"
#include "model.h"
#include "conjugate.h"
#include "sparse_conjugate.h"
#include "forward.h"
#include "inverse.h"
#include "solver.h"

#endif
