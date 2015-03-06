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
// #define CUDA_ERROR_CHECK

#ifndef MPFLOW_INCLUDE_CUDA_ERROR_H
#define MPFLOW_INCLUDE_CUDA_ERROR_H

// enable cuda error checking in debug configuration
#ifdef DEBUG
    #define CUDA_ERROR_CHECK
#endif

#define CudaSafeCall(error)   __cudaSafeCall(error, #error, __FILE__, __LINE__)
#define CudaCheckError()    __cudaCheckError(__FILE__, __LINE__)

#ifndef CUDA_ERROR_CHECK

inline void __cudaSafeCall(cudaError, const char*, const char*, const int) { }
inline void __cudaCheckError(const char*, const int) { }

#else

#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

inline void __cudaSafeCall(cudaError error, const char* function, const char* file, const int line) {
    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << file << ":" << line << " " << function;

        throw thrust::system_error(error, thrust::cuda_category(), ss.str());
    }
}

inline void __cudaCheckError(const char* file, const int line) {
    // More careful checking. However, this will affect performance.
    cudaDeviceSynchronize();
    cudaError error = cudaGetLastError();

    if (error != cudaSuccess) {
        std::stringstream ss;
        ss << file << ":" << line;

        throw thrust::system_error(error, thrust::cuda_category(), ss.str());
    }
}

#endif

#endif
