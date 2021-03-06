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
// Copyright (C) 2015 Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de
// --------------------------------------------------------------------

#ifndef MPFLOW_INCLUDE_MPFLOW_H
#define MPFLOW_INCLUDE_MPFLOW_H

#include "common.h"
#include "version.h"
#include "cuda_error.h"
#include "constants.h"
#include "mathematics.h"
#include "type_traits.h"
#include "json_helper.h"

// numeric solver
#include "numeric/cublas_wrapper.h"
#include "numeric/constants.h"
#include "numeric/matrix.h"
#include "numeric/sparse_matrix.h"
#include "numeric/irregular_mesh.h"
#include "numeric/conjugate_gradient.h"
#include "numeric/bicgstab.h"
#include "numeric/cpu_solver.h"
#include "numeric/preconditioner.h"

// FEM specific
#include "fem/basis.h"
#include "fem/ports.h"
#include "fem/sources.h"
#include "fem/equation.h"

// Models
#include "models/eit.h"
#include "models/mwi.h"
#include "models/constant.h"

// generic solver
#include "solver/inverse.h"
#include "solver/solver.h"

#endif
