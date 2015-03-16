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
#include "constants.h"
#include "mathematics.h"

// numeric solver
#include "numeric/cublas_wrapper.h"
#include "numeric/constants.h"
#include "numeric/matrix.h"
#include "numeric/sparse_matrix.h"
#include "numeric/irregular_mesh.h"
#include "numeric/conjugate_gradient.h"
#include "numeric/bicgstab.h"

// FEM specific
#include "fem/basis.h"
#include "fem/boundary_descriptor.h"
#include "fem/source_descriptor.h"
#include "fem/equation.h"

// generic solver
#include "solver/inverse.h"

// MWI specific
#include "mwi/equation.h"
#include "mwi/solver.h"

// EIT specific
#include "eit/forward.h"
#include "eit/solver.h"

#endif
