// --------------------------------------------------------------------
// This file is part of mpFlow.
//
// mpFlow is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
//
// mpFlow is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with mpFlow. If not, see <http://www.gnu.org/licenses/>.
//
// Copyright (C) 2014 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLUDE_MPFLOW_H
#define MPFLOW_INCLUDE_MPFLOW_H

#include "common.h"
#include "version.h"
#include "cuda_error.h"
#include "dtype.h"
#include "constants.h"
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
#include "uwb/model.h"

// basic solver
#include "solver/inverse.h"
#include "solver/solver.h"

#endif
