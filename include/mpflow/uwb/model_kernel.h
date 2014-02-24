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

#ifndef MPFLOW_INCLDUE_UWB_MODEL_KERNEL_H
#define MPFLOW_INCLDUE_UWB_MODEL_KERNEL_H

// namespaces mpFlow::UWB::modelKernel
namespace mpFlow {
namespace UWB {
namespace modelKernel {
    // reduce connectivity and elemental matrices
    template <
        class type
    >
    void reduceMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const type* intermediateMatrix, const dtype::index* columnIds,
        dtype::size rows, dtype::index offset, type* matrix);

    // update matrix kernel
    void updateMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const dtype::index* connectivityMatrix, const dtype::real* elementalMatrix,
        const dtype::real* material, dtype::size rows, dtype::size columns,
        dtype::real* matrixValues);

    // update system matrix kernel
    void updateSystemMatrix(dim3 blocks, dim3 threads, cudaStream_t stream,
        const mpFlow::dtype::real* sMatrixValues, const mpFlow::dtype::real* rMatrixValues,
        const mpFlow::dtype::index* sMatrixColumnIds, mpFlow::dtype::size density,
        mpFlow::dtype::real sScalar, mpFlow::dtype::real rScalar,
        mpFlow::dtype::index rowOffset, mpFlow::dtype::index columnOffset,
        mpFlow::dtype::real* systemMatrixValues);
}
}
}

#endif
