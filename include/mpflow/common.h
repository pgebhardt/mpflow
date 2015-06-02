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

#ifndef MPFLOW_INCLUDE_COMMON_H
#define MPFLOW_INCLUDE_COMMON_H

// std lib includes
#include <sys/stat.h>
#include <stdexcept>
#include <string>
#include <istream>
#include <tuple>
#include <array>
#include <vector>
#include <memory>
#include <algorithm>

// cuda includes
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <thrust/complex.h>
#include <complex>

// eigen for easier array handling on cpu
#include <Eigen/Dense>

// library for improved handling of strings
#include <stringtools/all.hpp>

// library from mesh generation
#include <distmesh/distmesh.h>

// library to parse json
#include "json.h"

#endif
