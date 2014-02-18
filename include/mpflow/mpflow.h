// mpFlow
//
// Copyright (C) 2014  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef MPFLOW_INCLUDE_MPFLOW_H
#define MPFLOW_INCLUDE_MPFLOW_H

#include "common.h"
#include "version.h"
#include "cuda_error.h"
#include "dtype.h"
#include "mathematics.h"

// numeric solver
#include "numeric/constants.h"
#include "numeric/matrix.h"
#include "numeric/sparse_matrix.h"
#include "numeric/irregular_mesh.h"
#include "numeric/conjugate.h"
#include "numeric/fast_conjugate.h"
#include "numeric/pre_conjugate.h"
#include "numeric/sparse_conjugate.h"

// FEM specific
#include "fem/basis.h"

// EIT specific
#include "eit/electrodes.h"
#include "eit/source.h"
#include "eit/model.h"
#include "eit/forward.h"

// UWB specific
#include "uwb/windows.h"

// basic solver
#include "solver/inverse.h"
#include "solver/solver.h"

#endif
