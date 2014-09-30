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

#ifndef MPFLOW_INCLDUE_EIT_FORWARD_KERNEL_H
#define MPFLOW_INCLDUE_EIT_FORWARD_KERNEL_H

namespace mpFlow {
namespace EIT {
namespace forwardKernel {
    void applyMixedBoundaryCondition(dim3 blocks, dim3 threads, cudaStream_t stream,
        dtype::real* excitation, const dtype::index* columnIds,
        dtype::real* values, dtype::index rows);
}
}
}

#endif
