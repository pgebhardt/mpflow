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

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "mpflow/cuda_error.h"

#include "mpflow/dtype.h"
#include "mpflow/numeric/constants.h"
#include "mpflow/eit/forward_kernel.h"

// calc voltage kernel
static __global__ void applyMeasurementPatternKernel(const mpFlow::dtype::real* potential,
    mpFlow::dtype::size offset, mpFlow::dtype::size rows, const mpFlow::dtype::real* pattern,
    mpFlow::dtype::size pattern_rows, bool additiv,
    mpFlow::dtype::real* voltage, mpFlow::dtype::size voltage_rows) {
    // get ids
    mpFlow::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    mpFlow::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc voltage
    mpFlow::dtype::real value = 0.0f;
    for (mpFlow::dtype::index electrode = 0; electrode < pattern_rows; ++electrode) {
        value += pattern[electrode + pattern_rows * row] * potential[offset + electrode + column * rows];
    }

    // set voltage
    if (additiv == true) {
        voltage[row + voltage_rows * column] += value;
    } else {
        voltage[row + voltage_rows * column] = value;
    }
}

// calc voltage kernel wrapper
void mpFlow::EIT::forwardKernel::applyMeasurementPattern(dim3 blocks, dim3 threads, cudaStream_t stream,
    const mpFlow::dtype::real* potential, mpFlow::dtype::size offset,
    mpFlow::dtype::size rows, const mpFlow::dtype::real* pattern,
    mpFlow::dtype::size pattern_rows, bool additiv,
    mpFlow::dtype::real* voltage, mpFlow::dtype::size voltage_rows) {
    // call cuda kernel
    applyMeasurementPatternKernel<<<blocks, threads, 0, stream>>>(
        potential, offset, rows, pattern, pattern_rows, additiv, voltage, voltage_rows);

    CudaCheckError();
}

