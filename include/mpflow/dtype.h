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

#ifndef MPFLOW_INCLUDE_DTYPE_H
#define MPFLOW_INCLUDE_DTYPE_H

// namespace mpFlow::dtype
namespace mpFlow {
namespace dtype {
    // basic scalar types
#ifdef USE_DOUBLE
    typedef double real;
#else
    typedef float real;
#endif
    typedef unsigned int size;
    typedef thrust::complex<real> complex;
    typedef unsigned int index;

    // invalid index
    static const index invalid_index = (index)(-1);
}
}

#endif
