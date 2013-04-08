// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#include <cstdlib>
#include <cstdio>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "fasteit/cuda_error.h"

#include "fasteit/dtype.h"
#include "fasteit/constants.h"
#include "fasteit/forward_kernel.h"

// calc voltage kernel
static __global__ void applyMeasurementPatternKernel(const fastEIT::dtype::real* potential,
    fastEIT::dtype::size offset, fastEIT::dtype::size rows, const fastEIT::dtype::real* pattern,
    fastEIT::dtype::size pattern_rows, bool additiv,
    fastEIT::dtype::real* voltage, fastEIT::dtype::size voltage_rows) {
    // get ids
    fastEIT::dtype::index row = blockIdx.x * blockDim.x + threadIdx.x;
    fastEIT::dtype::index column = blockIdx.y * blockDim.y + threadIdx.y;

    // calc voltage
    fastEIT::dtype::real value = 0.0f;
    for (fastEIT::dtype::index electrode = 0; electrode < pattern_rows; ++electrode) {
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
void fastEIT::forwardKernel::applyMeasurementPattern(dim3 blocks, dim3 threads, cudaStream_t stream,
    const fastEIT::dtype::real* potential, fastEIT::dtype::size offset,
    fastEIT::dtype::size rows, const fastEIT::dtype::real* pattern,
    fastEIT::dtype::size pattern_rows, bool additiv,
    fastEIT::dtype::real* voltage, fastEIT::dtype::size voltage_rows) {
    // call cuda kernel
    applyMeasurementPatternKernel<<<blocks, threads, 0, stream>>>(
        potential, offset, rows, pattern, pattern_rows, additiv, voltage, voltage_rows);

    CudaCheckError();
}

