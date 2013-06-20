// mpFlow
//
// Copyright (C) 2013  Patrik Gebhardt
// Contact: patrik.gebhardt@rub.de

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
