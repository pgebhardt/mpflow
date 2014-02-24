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

// namespace mpFlow::EIT::forwardKernel
namespace mpFlow {
namespace EIT {
namespace forwardKernel {
    // apply measurment pattern
    void applyMeasurementPattern(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dtype::real* potential, dtype::size offset,
        dtype::size rows, const dtype::real* pattern,
        dtype::size pattern_rows, bool additiv, dtype::real* voltage, dtype::size voltage_rows);
}
}
}

#endif
