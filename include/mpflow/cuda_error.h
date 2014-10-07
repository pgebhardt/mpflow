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

// Define this to turn on error checking
#define CUDA_ERROR_CHECK

#ifndef MPFLOW_INCLUDE_CUDA_ERROR_H
#define MPFLOW_INCLUDE_CUDA_ERROR_H

#define CudaSafeCall(err)   __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError()    __cudaCheckError(__FILE__, __LINE__)

#ifndef CUDA_ERROR_CHECK

inline void __cudaSafeCall(cudaError, const char*, const int) { }
inline void __cudaCheckError(const char*, const int) { }

#else

#include <cstdio>

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void __cudaCheckError( const char *file, const int line ) {
    // More careful checking. However, this will affect performance.
    // Comment away if needed.
    cudaThreadSynchronize();
    cudaError err = cudaGetLastError();

    if(cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n",
                 file, line, cudaGetErrorString(err));
        exit(-1);
    }
}

#endif

#endif
