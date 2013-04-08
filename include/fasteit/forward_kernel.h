// fastEIT
//
// Copyright (C) 2012  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

#ifndef FASTEIT_INCLUDE_FORWARD_KERNEL_H
#define FASTEIT_INCLUDE_FORWARD_KERNEL_H

// namespace fastEIT
namespace fastEIT {
    // namespace forward
    namespace forwardKernel {
        // apply measurment pattern
        void applyMeasurementPattern(dim3 blocks, dim3 threads, cudaStream_t stream,
            const dtype::real* potential, dtype::size offset,
            dtype::size rows, const dtype::real* pattern,
            dtype::size pattern_rows, bool additiv, dtype::real* voltage, dtype::size voltage_rows);
    }
}

#endif
